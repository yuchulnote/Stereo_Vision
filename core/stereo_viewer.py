# -*- coding: utf-8 -*-
"""
스테레오 비전 뷰어 (Consumer/Inference)
두 카메라의 공유 메모리 데이터를 읽어와 화면에 표시
"""

import cv2
import numpy as np
import multiprocessing as mp
import time
import threading
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import sys
from enum import Enum, auto

import os
from datetime import datetime
import yaml

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_buffer import SharedRingBuffer
from utils.logger import get_logger
from utils.camera_config import load_config
from model.pose_detector import PoseDetector
from calibration.temporal_sync import TemporalSynchronizer

import mediapipe as mp_lib

class SyncState(Enum):
    IDLE = auto()
    WAITING_FOR_GESTURE = auto() # 사용자에게 동작 유도
    COUNTDOWN = auto()           # 제스처 감지 후 카운트다운
    BUFFERING = auto()           # 데이터 수집 중
    CALCULATING = auto()         # 동기화 계산 중
    SYNCED = auto()              # 동기화 완료
    FAILED = auto()              # 동기화 실패

class StereoViewer:
    """스테레오 비전 뷰어 클래스"""
    
    def __init__(
        self,
        buffer_0: SharedRingBuffer,
        buffer_1: SharedRingBuffer,
        config: Optional[Dict] = None,
        config_path: str = "config.yaml"
    ):
        """
        Args:
            buffer_0: 카메라 0의 공유 메모리 버퍼
            buffer_1: 카메라 1의 공유 메모리 버퍼
            config: 설정 딕셔너리
            config_path: 설정 파일 경로
        """
        self.buffer_0 = buffer_0
        self.buffer_1 = buffer_1
        
        if config is None:
            try:
                config = load_config(config_path)
            except Exception:
                config = {}
        
        self.config = config
        
        self.running = False
        self.logger = get_logger()
        
        # FPS 측정
        self.fps_0 = 0.0
        self.fps_1 = 0.0
        self.fps_combined = 0.0
        
        self.frame_count_0 = 0
        self.frame_count_1 = 0
        self.frame_count_combined = 0
        
        self.last_fps_time = time.time()
        
        # 동기화 오차 측정
        self.sync_errors = []
        
        # 비디오 녹화
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording = False
        
        # CPU 모니터링
        self.cpu_monitor_thread: Optional[threading.Thread] = None
        self.cpu_usage = 0.0
        
        # 포즈 감지기 초기화
        self.use_pose_estimation = self.config.get('pose_estimation', {}).get('enabled', True)
        if self.use_pose_estimation:
            self.logger.info("MediaPipe 포즈 감지기 초기화 중...")
            model_complexity = self.config.get('pose_estimation', {}).get('model_complexity', 1)
            use_cuda = self.config.get('pose_estimation', {}).get('use_cuda', False)
            
            self.detector_0 = PoseDetector(
                model_complexity=model_complexity,
                use_cuda=use_cuda
            )
            self.detector_1 = PoseDetector(
                model_complexity=model_complexity,
                use_cuda=use_cuda
            )
            self.logger.info("MediaPipe 포즈 감지기 초기화 완료")
        
        # --- 시간 동기화 (Temporal Sync) 관련 초기화 ---
        self.sync_state = SyncState.IDLE
        self.synchronizer = TemporalSynchronizer(fps=30.0) # FPS는 나중에 업데이트됨
        self.sync_buffer_0: List[float] = [] # Y좌표 저장
        self.sync_buffer_1: List[float] = []
        self.sync_start_time = 0.0
        self.countdown_start_time = 0.0
        self.calculated_time_offset = 0.0 # ms 단위
        self.last_sync_result = ""
        self.sync_graphs = (None, None) # 시각화용 데이터
        
        # --- 캘리브레이션(Calibration) 관련 초기화 ---
        self.calibration_mode = False
        self.chessboard_size = (8, 5) # 내부 코너 개수 (가로-1, 세로-1) -> 사각형 9x6 기준
        self.capture_dir = Path(project_root) / "data" / "calibration_images"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.saved_count = 0
        # 기존 저장된 파일 개수 확인
        existing_files = list(self.capture_dir.glob("left_*.jpg"))
        if existing_files:
            self.saved_count = len(existing_files)
            
        # --- 정류(Rectification) 관련 초기화 ---
        self.rectification_mode = False
        self.calib_data = None
        self.map_l1, self.map_l2 = None, None
        self.map_r1, self.map_r2 = None, None
        
        # 캘리브레이션 결과 로드 시도
        self.load_calibration()
    
    def load_calibration(self, path="calibration_result.yaml"):
        """캘리브레이션 결과 파일 로드 및 Rectification Map 생성"""
        if not os.path.exists(path):
            self.logger.warning(f"캘리브레이션 결과 파일이 없습니다: {path}")
            return False
            
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                
            # 데이터 로드 (List -> Numpy array 변환)
            mtx_l = np.array(data['camera_matrix_left'])
            dist_l = np.array(data['dist_coeffs_left'])
            mtx_r = np.array(data['camera_matrix_right'])
            dist_r = np.array(data['dist_coeffs_right'])
            R = np.array(data['R'])
            T = np.array(data['T'])
            image_size = tuple(data['image_size']) # (width, height)
            
            # Stereo Rectification
            # 두 이미지를 평행하게 만드는 회전/투영 행렬 계산
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                mtx_l, dist_l, mtx_r, dist_r, image_size, R, T
            )
            
            # 매핑 테이블 생성 (이걸 미리 만들어둬야 실시간 처리가 빠름)
            # m1type=cv2.CV_16SC2 (고정 소수점)가 CV_32FC1보다 빠름
            self.map_l1, self.map_l2 = cv2.initUndistortRectifyMap(
                mtx_l, dist_l, R1, P1, image_size, cv2.CV_16SC2
            )
            self.map_r1, self.map_r2 = cv2.initUndistortRectifyMap(
                mtx_r, dist_r, R2, P2, image_size, cv2.CV_16SC2
            )
            
            self.calib_data = data
            self.logger.info("캘리브레이션 데이터 로드 및 Rectification Map 생성 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"캘리브레이션 데이터 로드 실패: {e}")
            return False

    def start_recording(self, output_path: str, fps: int = 30, codec: str = "XVID"):
        """비디오 녹화 시작"""
        try:
            # 해상도 확인
            width = self.buffer_0.width
            height = self.buffer_0.height
            
            # 결합된 영상 크기 (좌우)
            combined_width = width * 2
            combined_height = height
            
            # 비디오 라이터 생성
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.video_writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (combined_width, combined_height)
            )
            
            if not self.video_writer.isOpened():
                self.logger.error(f"비디오 녹화 시작 실패: {output_path}")
                return False
            
            self.recording = True
            self.logger.info(f"비디오 녹화 시작: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"비디오 녹화 시작 중 오류: {e}")
            return False
    
    def stop_recording(self):
        """비디오 녹화 중지"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.recording = False
        self.logger.info("비디오 녹화 중지")
    
    def monitor_cpu(self):
        """CPU 사용량 모니터링 (별도 스레드)"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        while self.running:
            try:
                # CPU 사용률 (1초 간격)
                cpu_percent = process.cpu_percent(interval=1.0)
                self.cpu_usage = cpu_percent
                
                # 전체 시스템 CPU 사용률 (코어별)
                try:
                    cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
                    self.logger.debug(
                        f"프로세스 CPU 사용률: {cpu_percent:.1f}% "
                        f"(시스템 코어별: {[f'{c:.1f}%' for c in cpu_per_core]})"
                    )
                except Exception:
                    # percpu가 지원되지 않는 경우 무시
                    self.logger.debug(f"프로세스 CPU 사용률: {cpu_percent:.1f}%")
                
            except Exception as e:
                self.logger.error(f"CPU 모니터링 오류: {e}")
                break
    
    def read_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        """
        두 카메라에서 프레임 읽기
        
        Returns:
            (frame_0, frame_1, timestamp_0_ns, timestamp_1_ns) 또는 None
        """
        # 두 버퍼에서 프레임 읽기
        result_0 = self.buffer_0.read_frame()
        result_1 = self.buffer_1.read_frame()
        
        # 디버깅: 프레임 읽기 상태 확인
        if not hasattr(self, '_read_debug_count'):
            self._read_debug_count = 0
            self._last_debug_time = time.time()
        
        current_time = time.time()
        # 1초마다 디버깅 정보 출력
        if current_time - self._last_debug_time >= 1.0:
            self._last_debug_time = current_time
            self._read_debug_count += 1
            
            if self._read_debug_count <= 10:  # 처음 10초 동안만
                if result_0 is None:
                    self.logger.warning(f"버퍼 0에서 프레임 읽기 실패 (시도 {self._read_debug_count})")
                else:
                    self.logger.info(f"버퍼 0에서 프레임 읽기 성공")
                    
                if result_1 is None:
                    self.logger.warning(f"버퍼 1에서 프레임 읽기 실패 (시도 {self._read_debug_count})")
                else:
                    self.logger.info(f"버퍼 1에서 프레임 읽기 성공")
        
        # 둘 다 None이면 프레임 없음
        if result_0 is None and result_1 is None:
            return None
        
        # 하나만 None인 경우, 스테레오 비전을 위해서는 두 프레임이 모두 필요
        if result_0 is None or result_1 is None:
            # 하나의 버퍼만 비어있는 경우, None 반환
            return None
        
        frame_0, ts_0, fn_0 = result_0
        frame_1, ts_1, fn_1 = result_1
        
        # --- Soft-Genlock (Step 22) ---
        # 계산된 오차만큼 ts_1 보정 (단위: ns)
        # Offset이 양수(Lag)면 ts_1이 늦는 것이므로 ts_1을 앞당기거나(빼기) 
        # 상대적 비교를 위해 보정. 
        # 여기서는 오차 분석을 위해 원본 타임스탬프를 보정해서 반환
        ts_1_corrected = ts_1 - int(self.calculated_time_offset * 1_000_000)
        
        # 동기화 오차 계산 (Step 17)
        sync_error_ns = abs(ts_0 - ts_1_corrected)
        sync_error_ms = sync_error_ns / 1_000_000.0
        
        self.sync_errors.append(sync_error_ms)
        if len(self.sync_errors) > 100:
            self.sync_errors.pop(0)
        
        return frame_0, frame_1, ts_0, ts_1_corrected
    
    def update_fps(self):
        """FPS 업데이트"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps_0 = self.frame_count_0 / elapsed
            self.fps_1 = self.frame_count_1 / elapsed
            self.fps_combined = self.frame_count_combined / elapsed
            
            # Synchronizer FPS 업데이트
            if self.fps_combined > 0:
                self.synchronizer.fps = self.fps_combined
            
            self.frame_count_0 = 0
            self.frame_count_1 = 0
            self.frame_count_combined = 0
            self.last_fps_time = current_time
            
            # 로거에 FPS 업데이트
            self.logger.update_fps(self.fps_combined)
            
            # 평균 동기화 오차
            if self.sync_errors:
                avg_sync_error = sum(self.sync_errors) / len(self.sync_errors)
                self.logger.update_sync_error(avg_sync_error)
    
    def draw_info(self, frame: np.ndarray, camera_idx: int, fps: float):
        """프레임에 정보 표시"""
        # 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 텍스트
        cv2.putText(frame, f"Camera {camera_idx}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CPU: {self.cpu_usage:.1f}%", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.sync_errors:
            avg_sync = sum(self.sync_errors[-10:]) / len(self.sync_errors[-10:])
            color = (0, 255, 0) if avg_sync < 10.0 else (0, 0, 255)
            cv2.putText(frame, f"SyncErr: {avg_sync:.2f}ms", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 동기화 상태 표시
        cv2.putText(frame, f"State: {self.sync_state.name}", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        if self.calculated_time_offset != 0:
            cv2.putText(frame, f"Offset: {self.calculated_time_offset:.1f}ms", (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    def _extract_wrist_y(self, results) -> Optional[float]:
        """MediaPipe 결과에서 오른쪽 손목 Y좌표 추출"""
        if results and results.pose_landmarks:
            # RIGHT_WRIST = 16
            wrist = results.pose_landmarks.landmark[mp_lib.solutions.pose.PoseLandmark.RIGHT_WRIST]
            return wrist.y
        return None

    def _is_ready_pose(self, results) -> bool:
        """오른쪽 손목이 코보다 위에 있는지 확인 (제스처 감지)"""
        if results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_lib.solutions.pose.PoseLandmark.NOSE]
            wrist = landmarks[mp_lib.solutions.pose.PoseLandmark.RIGHT_WRIST]
            # y는 위쪽이 0이므로 작을수록 위에 있음
            # 확실한 의도를 위해 코보다 약간 더 위(0.05 여유)일 때 인식
            return wrist.y < (nose.y - 0.05)
        return False

    def draw_sync_guide(self, frame: np.ndarray):
        """동기화 가이드 UI 표시 (Step 11, Step 23)"""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        
        # 폰트 스케일 및 위치 비례 계산
        font_scale_large = min(w, h) / 600.0  # 기본 1.0 @ 600px
        font_scale_small = font_scale_large * 0.6
        y_offset_large = int(h * 0.1)
        y_offset_small = int(h * 0.05)
        
        if self.sync_state == SyncState.WAITING_FOR_GESTURE:
            msg1 = "RAISE RIGHT HAND"
            msg2 = "above your head"
            # 텍스트 사이즈 계산하여 중앙 정렬
            (fw, fh), _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 3)
            cv2.putText(frame, msg1, (cx - fw//2, cy - y_offset_small), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 165, 255), 3)
            
            (fw2, fh2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 2)
            cv2.putText(frame, msg2, (cx - fw2//2, cy + y_offset_large), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (255, 255, 255), 2)
            
        elif self.sync_state == SyncState.COUNTDOWN:
            elapsed = time.time() - self.countdown_start_time
            remaining = max(0.0, 1.5 - elapsed)
            msg = f"HOLD... {remaining:.1f}"
            (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large * 1.5, 3)
            cv2.putText(frame, msg, (cx - fw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large * 1.5, (0, 255, 255), 3)

        elif self.sync_state == SyncState.BUFFERING:
            elapsed = time.time() - self.sync_start_time
            if elapsed < 1.0:
                 msg = "DROP FAST!!!"
                 (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large * 2.0, 4)
                 cv2.putText(frame, msg, (cx - fw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large * 2.0, (0, 0, 255), 4)
            else:
                 msg = f"Recording... {elapsed:.1f}s"
                 (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)
                 cv2.putText(frame, msg, (cx - fw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 255), 2)
            
        elif self.sync_state == SyncState.CALCULATING:
            msg = "CALCULATING..."
            (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)
            cv2.putText(frame, msg, (cx - fw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 255), 2)

        elif self.sync_state == SyncState.SYNCED:
            # 동기화 결과 그래프 그리기
            if self.sync_graphs and self.sync_graphs[0] is not None:
                self._draw_signal_graph(frame, self.sync_graphs[0], self.sync_graphs[1])
            
            msg = f"SYNCED! Offset: {self.calculated_time_offset:.2f}ms"
            (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)
            # 그래프 위쪽으로 위치 조정
            cv2.putText(frame, msg, (cx - fw//2, cy - int(h * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 0), 2)
            
            msg2 = "Press 'S' to Retry"
            (fw2, fh2), _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 1)
            cv2.putText(frame, msg2, (cx - fw2//2, cy + int(h * 0.3)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), 1)

        elif self.sync_state == SyncState.FAILED:
             msg = "SYNC FAILED. RETRY."
             (fw, fh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)
             cv2.putText(frame, msg, (cx - fw//2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 0, 255), 2)

    def _draw_signal_graph(self, frame: np.ndarray, sig1: np.ndarray, sig2: np.ndarray):
        """두 신호 그래프 그리기 (OpenCV)"""
        h, w = frame.shape[:2]
        # 그래프 영역 설정 (화면 비율에 맞춤)
        graph_w = int(w * 0.4)  # 화면 너비의 40%
        graph_h = int(h * 0.25) # 화면 높이의 25%
        x_start = (w - graph_w) // 2
        y_start = (h - graph_h) // 2 + int(h * 0.1) # 화면 중앙보다 약간 아래
        
        # 배경 박스 (반투명)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_start + graph_w, y_start + graph_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 데이터 정규화 (그래프 높이에 맞춤)
        # sig1, sig2는 이미 Z-score 정규화되어 있음 (대략 -3 ~ 3 범위)
        # 이를 0 ~ graph_h 범위로 변환
        def normalize_for_plot(sig):
            if len(sig) == 0: return []
            min_val, max_val = -3.0, 3.0 # Z-score 예상 범위
            norm = (sig - min_val) / (max_val - min_val) # 0.0 ~ 1.0
            norm = np.clip(norm, 0.0, 1.0)
            return (norm * graph_h).astype(int)

        pts1 = normalize_for_plot(sig1)
        pts2 = normalize_for_plot(sig2)
        
        # 그리기
        if len(pts1) > 1 and len(pts2) > 1:
            # X축 간격
            step = graph_w / len(pts1)
            
            for i in range(len(pts1) - 1):
                x1 = int(x_start + i * step)
                x2 = int(x_start + (i + 1) * step)
                
                # Cam 1 (Yellow)
                y1_1 = int(y_start + graph_h - pts1[i])
                y1_2 = int(y_start + graph_h - pts1[i+1])
                cv2.line(frame, (x1, y1_1), (x2, y1_2), (0, 255, 255), 2)
                
                # Cam 2 (Magenta)
                y2_1 = int(y_start + graph_h - pts2[i])
                y2_2 = int(y_start + graph_h - pts2[i+1])
                cv2.line(frame, (x1, y2_1), (x2, y2_2), (255, 0, 255), 2)

        # 범례
        font_scale = min(w, h) / 1000.0
        cv2.putText(frame, "Cam 1 (Yellow)", (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)
        cv2.putText(frame, "Cam 2 (Magenta)", (x_start + int(graph_w * 0.5), y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), 1)

    def _process_calibration(self, frame_0, frame_1):
        """캘리브레이션 모드 처리: 체스보드 찾기 및 그리기"""
        # 그레이스케일 변환
        gray_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 찾기 (비용이 비싸므로 3프레임마다 하거나, 여기서 그냥 수행하고 느리면 조절)
        # CALIB_CB_FAST_CHECK 플래그를 사용하면 없을 때 빨리 리턴함
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ret_0, corners_0 = cv2.findChessboardCorners(gray_0, self.chessboard_size, flags)
        ret_1, corners_1 = cv2.findChessboardCorners(gray_1, self.chessboard_size, flags)
        
        # 찾았으면 그리기
        if ret_0:
            cv2.drawChessboardCorners(frame_0, self.chessboard_size, corners_0, ret_0)
        if ret_1:
            cv2.drawChessboardCorners(frame_1, self.chessboard_size, corners_1, ret_1)
            
        return ret_0 and ret_1 # 둘 다 찾았는지 여부 반환

    def _process_rectification(self, frame_0, frame_1):
        """이미지 정류 (Rectification) 및 수평선 그리기"""
        if self.map_l1 is None:
            return frame_0, frame_1
            
        # 1. Remap (왜곡 보정 + 평행 정렬)
        rect_0 = cv2.remap(frame_0, self.map_l1, self.map_l2, cv2.INTER_LINEAR)
        rect_1 = cv2.remap(frame_1, self.map_r1, self.map_r2, cv2.INTER_LINEAR)
        
        # 2. 수평선 그리기 (확인용)
        h, w = rect_0.shape[:2]
        for y in range(0, h, 30): # 30픽셀 간격
            cv2.line(rect_0, (0, y), (w, y), (0, 255, 0), 1)
            cv2.line(rect_1, (0, y), (w, y), (0, 255, 0), 1)
            
        return rect_0, rect_1

    def run(self):
        """뷰어 실행"""
        self.running = True
        
        # CPU 모니터링 스레드 시작
        try:
            import psutil
            self.cpu_monitor_thread = threading.Thread(target=self.monitor_cpu, daemon=True)
            self.cpu_monitor_thread.start()
        except ImportError:
            self.logger.warning("psutil이 설치되지 않아 CPU 모니터링을 사용할 수 없습니다.")
        
        self.logger.info("스테레오 뷰어 시작")
        
        # 초기 프레임 대기
        frame_wait_start = time.time()
        frame_wait_timeout = 30.0 
        
        try:
            while self.running:
                start_time = time.time()
                
                # 프레임 읽기
                result = self.read_frames()
                
                if result is None:
                    elapsed = time.time() - frame_wait_start
                    if elapsed > 5.0 and elapsed < 5.1:
                        self.logger.warning(f"프레임 대기 중... ({elapsed:.1f}s)")
                        frame_wait_start = time.time()
                    
                    if elapsed > frame_wait_timeout:
                        self.logger.error("프레임 읽기 타임아웃")
                        break
                    
                    time.sleep(0.01)
                    continue
                
                # 프레임을 읽었으면 대기 시간 리셋
                frame_wait_start = 0 
                
                frame_0, frame_1, ts_0, ts_1 = result
                
                # FPS 카운트
                self.frame_count_0 += 1
                self.frame_count_1 += 1
                self.frame_count_combined += 1
                
                self.update_fps()
                
                # 포즈 추정 (MediaPipe) & 데이터 수집
                wrist_y_0 = None
                wrist_y_1 = None
                
                if self.use_pose_estimation:
                    # 카메라 0
                    results_0 = self.detector_0.process(frame_0)
                    self.detector_0.draw_landmarks(frame_0, results_0)
                    wrist_y_0 = self._extract_wrist_y(results_0)
                    
                    # 카메라 1
                    results_1 = self.detector_1.process(frame_1)
                    self.detector_1.draw_landmarks(frame_1, results_1)
                    wrist_y_1 = self._extract_wrist_y(results_1)
                
                # --- 캘리브레이션 모드 ---
                calibration_ready = False
                if self.calibration_mode:
                    calibration_ready = self._process_calibration(frame_0, frame_1)
                    
                    # 캘리브레이션 상태 표시
                    h, w = frame_0.shape[:2]
                    status_color = (0, 255, 0) if calibration_ready else (0, 0, 255)
                    status_text = "READY TO CAPTURE (Press SPACE)" if calibration_ready else "SEARCHING..."
                    
                    # 각 프레임 상단에 표시
                    header_y = 30
                    cv2.putText(frame_0, "CALIBRATION MODE", (w//2 - 100, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame_1, "CALIBRATION MODE", (w//2 - 100, header_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 현재 패턴 사이즈 표시 (조절 안내)
                    cols, rows = self.chessboard_size
                    pattern_text = f"Corners: {cols}x{rows} (Squares: {cols+1}x{rows+1})"
                    cv2.putText(frame_0, pattern_text, (20, header_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(frame_1, pattern_text, (20, header_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    cv2.putText(frame_0, status_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    cv2.putText(frame_1, status_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # --- 정류(Rectification) 모드 ---
                elif self.rectification_mode:
                    if self.calib_data is not None:
                        frame_0, frame_1 = self._process_rectification(frame_0, frame_1)
                        
                        # 상태 표시
                        msg = "RECTIFICATION MODE"
                        cv2.putText(frame_0, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame_1, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        msg = "NO CALIBRATION DATA"
                        cv2.putText(frame_0, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame_1, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # --- 동기화 로직 (State Machine) ---
                elif self.sync_state == SyncState.WAITING_FOR_GESTURE:
                    # 제스처 감지 (양쪽 카메라 중 하나라도 감지되면 OK)
                    ready_0 = self._is_ready_pose(results_0) if self.use_pose_estimation else False
                    ready_1 = self._is_ready_pose(results_1) if self.use_pose_estimation else False
                    
                    if ready_0 or ready_1:
                        self.sync_state = SyncState.COUNTDOWN
                        self.countdown_start_time = time.time()
                        self.logger.info("제스처 감지됨! 카운트다운 시작")

                elif self.sync_state == SyncState.COUNTDOWN:
                    elapsed = time.time() - self.countdown_start_time
                    if elapsed > 1.5: # 1.5초 대기 후 시작
                        self.sync_state = SyncState.BUFFERING
                        self.sync_start_time = time.time()
                        self.sync_buffer_0 = []
                        self.sync_buffer_1 = []
                        self.logger.info("데이터 버퍼링 시작...")

                elif self.sync_state == SyncState.BUFFERING:
                    # Step 12: 버퍼링
                    if wrist_y_0 is not None and wrist_y_1 is not None:
                        self.sync_buffer_0.append(wrist_y_0)
                        self.sync_buffer_1.append(wrist_y_1)
                    else:
                        # 놓친 프레임은 이전 값이나 0으로 채우기 보단 일단 건너뛰거나 보간 필요
                        # 간단하게는 None 처리 로직이 필요하지만 여기선 데이터가 있는 경우만 수집
                        pass
                    
                    # 4초(약 120프레임 @ 30fps) 수집
                    if time.time() - self.sync_start_time > 4.0:
                        self.sync_state = SyncState.CALCULATING
                
                elif self.sync_state == SyncState.CALCULATING:
                    # Step 13~20: 동기화 계산
                    if len(self.sync_buffer_0) > 30 and len(self.sync_buffer_1) > 30:
                        self.logger.info("동기화 계산 시작...")
                        offset, corr, processed_signals = self.synchronizer.calculate_time_offset(
                            self.sync_buffer_0, self.sync_buffer_1
                        )
                        
                        # Step 21: 유효 범위 검증 (예: ±500ms)
                        if abs(offset) < 500.0 and corr > 0.3: # 상관계수 임계값은 조정 필요
                            self.calculated_time_offset = offset
                            self.sync_graphs = processed_signals
                            self.sync_state = SyncState.SYNCED
                            self.logger.info(f"동기화 성공! Offset: {offset:.2f}ms")
                        else:
                            self.sync_state = SyncState.FAILED
                            self.logger.warning(f"동기화 실패. Offset: {offset:.2f}ms, Corr: {corr:.3f}")
                    else:
                        self.sync_state = SyncState.FAILED
                        self.logger.warning("데이터 부족으로 동기화 실패")
                
                # UI 표시
                self.draw_info(frame_0, 0, self.fps_0)
                self.draw_info(frame_1, 1, self.fps_1)
                
                # 좌우 결합
                combined_frame = np.hstack([frame_0, frame_1])
                
                # 가이드 표시 (전체 화면 중앙)
                self.draw_sync_guide(combined_frame)
                
                # 비디오 녹화
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(combined_frame)
                
                # 화면 표시
                try:
                    display_frame = combined_frame
                    if display_frame.shape[1] > 1600:
                        scale = 1900 / display_frame.shape[1]
                        new_width = 1900
                        new_height = int(display_frame.shape[0] * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow('Stereo Vision', display_frame)
                except Exception as e:
                    self.logger.error(f"화면 표시 오류: {e}")
                    break
                
                latency_ms = (time.time() - start_time) * 1000
                self.logger.update_latency(latency_ms)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 정류 모드 토글 (녹화 키는 'v'로 변경 예정 또는 임시 비활성화)
                    if self.calib_data is None:
                        # 데이터가 없으면 로드 시도
                        self.load_calibration()
                    
                    self.rectification_mode = not self.rectification_mode
                    self.logger.info(f"정류 모드 {'시작' if self.rectification_mode else '종료'}")
                
                elif key == ord('v'): # Video Recording
                    if not self.recording:
                        output_path = self.config.get('output', {}).get('video_path', 'output/stereo_output.avi')
                        self.start_recording(output_path)
                    else:
                        self.stop_recording()
                        
                elif key == ord('s'): # Sync 시작 진입
                    if not self.calibration_mode:
                        self.sync_state = SyncState.WAITING_FOR_GESTURE
                        self.sync_buffer_0 = []
                        self.sync_buffer_1 = []
                        self.logger.info("동기화 모드 진입: 사용자 제스처 대기")
                elif key == ord('c'): # 캘리브레이션 모드 토글
                    self.calibration_mode = not self.calibration_mode
                    self.logger.info(f"캘리브레이션 모드 {'시작' if self.calibration_mode else '종료'}")
                    
                elif key == ord(' '):
                    if self.calibration_mode:
                        # 캘리브레이션 이미지 캡처
                        if calibration_ready:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            path_0 = self.capture_dir / f"left_{timestamp}.jpg"
                            path_1 = self.capture_dir / f"right_{timestamp}.jpg"
                            
                            cv2.imwrite(str(path_0), frame_0)
                            cv2.imwrite(str(path_1), frame_1)
                            
                            self.saved_count += 1
                            self.logger.info(f"캘리브레이션 이미지 저장 완료 ({self.saved_count}쌍): {path_0.name}")
                            
                            # 저장 확인 UI (잠시 반짝이거나 메시지)
                            # 여기서는 간단히 로그로 대체하고 화면 깜빡임은 구현 복잡도상 생략
                        else:
                            self.logger.warning("체스보드가 감지되지 않아 캡처할 수 없습니다.")
                    
                    elif self.sync_state == SyncState.WAITING_FOR_GESTURE:
                        # 수동 트리거 (백업용)
                        self.sync_state = SyncState.BUFFERING
                    self.sync_start_time = time.time()
                    self.logger.info("데이터 버퍼링 시작 (수동 트리거)...")
                
                # 체스보드 크기 조절 (화살표 키)
                elif self.calibration_mode:
                    cols, rows = self.chessboard_size
                    if key == 82: # Up Arrow (OpenCV waitKey code may vary by platform, trying standard)
                        self.chessboard_size = (cols, rows + 1)
                    elif key == 84: # Down Arrow
                        self.chessboard_size = (cols, max(3, rows - 1))
                    elif key == 83: # Right Arrow
                        self.chessboard_size = (cols + 1, rows)
                    elif key == 81: # Left Arrow
                        self.chessboard_size = (max(3, cols - 1), rows)
                    
                    # 윈도우/리눅스 호환을 위해 확장 키 코드 처리 (255, 0 등)가 필요할 수 있으므로
                    # 간단하게 wasd 키도 지원
                    if key == ord('w'): self.chessboard_size = (cols, rows + 1)
                    elif key == ord('s'): self.chessboard_size = (cols, max(3, rows - 1))
                    elif key == ord('d'): self.chessboard_size = (cols + 1, rows)
                    elif key == ord('a'): self.chessboard_size = (max(3, cols - 1), rows)
                    
                    if self.chessboard_size != (cols, rows):
                        self.logger.info(f"체스보드 패턴 변경: {self.chessboard_size}")
                
        except KeyboardInterrupt:
            self.logger.info("뷰어 중단 요청")
        except Exception as e:
            self.logger.error(f"뷰어 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리 작업"""
        self.running = False
        self.stop_recording()
        
        # 감지기 리소스 해제
        if hasattr(self, 'detector_0'):
            self.detector_0.close()
        if hasattr(self, 'detector_1'):
            self.detector_1.close()
            
        cv2.destroyAllWindows()
        self.logger.info("스테레오 뷰어 정리 완료")
