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
from typing import Optional, Tuple, Dict
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_buffer import SharedRingBuffer
from utils.logger import get_logger
from utils.camera_config import load_config


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
        
        # 동기화 오차 계산 (Step 17)
        sync_error_ns = abs(ts_0 - ts_1)
        sync_error_ms = sync_error_ns / 1_000_000.0
        
        self.sync_errors.append(sync_error_ms)
        if len(self.sync_errors) > 100:
            self.sync_errors.pop(0)
        
        return frame_0, frame_1, ts_0, ts_1
    
    def update_fps(self):
        """FPS 업데이트"""
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.fps_0 = self.frame_count_0 / elapsed
            self.fps_1 = self.frame_count_1 / elapsed
            self.fps_combined = self.frame_count_combined / elapsed
            
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
        height, width = frame.shape[:2]
        
        # 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
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
            cv2.putText(frame, f"Sync: {avg_sync:.2f}ms", (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
        
        # 초기 프레임 대기 (Producer가 프레임을 쓰기 시작할 때까지)
        self.logger.info("프레임 대기 중...")
        frame_wait_start = time.time()
        frame_wait_timeout = 30.0  # 30초 타임아웃으로 증가 (초기화 지연 고려)
        
        try:
            while self.running:
                start_time = time.time()
                
                # 프레임 읽기
                result = self.read_frames()
                
                if result is None:
                    # 프레임이 없을 때 디버깅 정보
                    elapsed = time.time() - frame_wait_start
                    if elapsed > 5.0 and elapsed < 5.1:  # 5초마다 한 번씩만 로그
                        self.logger.warning(
                            f"프레임을 읽지 못하고 있습니다. "
                            f"대기 시간: {elapsed:.1f}초"
                        )
                        frame_wait_start = time.time()  # 리셋
                    
                    if elapsed > frame_wait_timeout:
                        self.logger.error(
                            f"프레임 읽기 타임아웃 ({frame_wait_timeout}초). "
                            f"카메라 Producer가 정상적으로 작동하는지 확인하세요."
                        )
                        break
                    
                    time.sleep(0.01)  # 버퍼가 비어있으면 잠시 대기 (1ms -> 10ms로 증가)
                    continue
                
                # 프레임을 읽었으면 대기 시간 리셋
                if frame_wait_start > 0:
                    elapsed = time.time() - frame_wait_start
                    if elapsed > 0.1:
                        self.logger.info(f"첫 프레임 수신 (대기 시간: {elapsed:.2f}초)")
                    frame_wait_start = 0  # 리셋
                
                frame_0, frame_1, ts_0, ts_1 = result
                
                # FPS 카운트
                self.frame_count_0 += 1
                self.frame_count_1 += 1
                self.frame_count_combined += 1
                
                # FPS 업데이트
                self.update_fps()
                
                # 정보 표시
                self.draw_info(frame_0, 0, self.fps_0)
                self.draw_info(frame_1, 1, self.fps_1)
                
                # 디버깅: 프레임 데이터 확인 (검은 화면 문제 해결)
                if self.frame_count_combined % 30 == 0:  # 30프레임마다 한 번씩
                    mean_0 = np.mean(frame_0)
                    mean_1 = np.mean(frame_1)
                    if mean_0 < 1.0 or mean_1 < 1.0:
                        self.logger.warning(
                            f"검은 화면 감지됨 - 평균값: "
                            f"Cam0={mean_0:.1f}, Cam1={mean_1:.1f}"
                        )
                
                # 좌우 결합 (Step 20)
                combined_frame = np.hstack([frame_0, frame_1])
                
                # 비디오 녹화 (Step 22)
                if self.recording and self.video_writer is not None:
                    self.video_writer.write(combined_frame)
                
                # 화면 표시
                try:
                    # 화면 표시용 리사이즈 (녹화는 원본 화질 유지)
                    # 전체 너비가 1600px를 넘으면 리사이즈
                    display_frame = combined_frame
                    if display_frame.shape[1] > 1600:
                        scale = 1600 / display_frame.shape[1]
                        new_width = 1600
                        new_height = int(display_frame.shape[0] * scale)
                        display_frame = cv2.resize(display_frame, (new_width, new_height))
                    
                    cv2.imshow('Stereo Vision', display_frame)
                except Exception as e:
                    self.logger.error(f"화면 표시 오류: {e}")
                    break
                
                # 처리 지연 시간 측정
                latency_ms = (time.time() - start_time) * 1000
                self.logger.update_latency(latency_ms)
                
                # 키 입력 처리 (waitKey는 화면 업데이트에도 필요)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # 녹화 토글
                    if not self.recording:
                        output_path = self.config.get('output', {}).get('video_path', 'output/stereo_output.avi')
                        self.start_recording(output_path)
                    else:
                        self.stop_recording()
                
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
        cv2.destroyAllWindows()
        self.logger.info("스테레오 뷰어 정리 완료")

