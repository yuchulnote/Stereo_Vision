# -*- coding: utf-8 -*-
"""
포즈 추론 프로세스 (별도 프로세스)
MediaPipe 추론을 별도 프로세스에서 수행하여 StereoViewer의 병목을 해소
"""

import cv2
import numpy as np
import multiprocessing as mp
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_buffer import SharedRingBuffer
from model.pose_detector import PoseDetector
from utils.logger import get_logger
from utils.camera_config import load_config


@dataclass
class PoseResult:
    """포즈 추론 결과 (multiprocessing.Queue를 통해 전달)"""
    wrist_y: Optional[float] = None
    ready_pose: bool = False
    timestamp_ns: int = 0
    frame_number: int = 0
    landmarks: Optional[list] = None  # 랜드마크 정보 (직렬화 가능한 형태)
    
    def __getstate__(self):
        """pickle을 위한 상태 반환"""
        return {
            'wrist_y': self.wrist_y,
            'ready_pose': self.ready_pose,
            'timestamp_ns': self.timestamp_ns,
            'frame_number': self.frame_number,
            'landmarks': self.landmarks
        }
    
    def __setstate__(self, state):
        """pickle에서 복원"""
        self.wrist_y = state['wrist_y']
        self.ready_pose = state['ready_pose']
        self.timestamp_ns = state['timestamp_ns']
        self.frame_number = state['frame_number']
        self.landmarks = state.get('landmarks', None)


class PoseInferenceProcess:
    """포즈 추론 프로세스 클래스"""
    
    def __init__(
        self,
        camera_index: int,
        input_buffer: SharedRingBuffer,
        output_queue: mp.Queue,
        config: Optional[Dict] = None,
        config_path: str = "config.yaml"
    ):
        """
        Args:
            camera_index: 카메라 인덱스 (식별용)
            input_buffer: 입력 프레임 버퍼
            output_queue: 추론 결과 출력 큐
            config: 설정 딕셔너리
            config_path: 설정 파일 경로
        """
        self.camera_index = camera_index
        self.input_buffer = input_buffer
        self.output_queue = output_queue
        self.config = config
        self.config_path = config_path
        
        self.running = mp.Value('b', False)
        
        # 로거 초기화
        try:
            if config is None:
                full_config = load_config(config_path) if config_path else {}
            else:
                full_config = config
            logger_config = full_config.get('logging', {})
            self.logger = get_logger(logger_config)
        except Exception:
            import logging
            self.logger = logging.getLogger(f'PoseInference-{camera_index}')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # 포즈 감지기 초기화
        self.use_pose_estimation = True
        if self.config:
            self.use_pose_estimation = self.config.get('pose_estimation', {}).get('enabled', True)
        
        if self.use_pose_estimation:
            model_complexity = self.config.get('pose_estimation', {}).get('model_complexity', 1) if self.config else 1
            use_cuda = self.config.get('pose_estimation', {}).get('use_cuda', False) if self.config else False
            
            self.detector = PoseDetector(
                model_complexity=model_complexity,
                use_cuda=use_cuda
            )
            self.logger.info(f"포즈 감지기 초기화 완료 (카메라 {self.camera_index})")
        else:
            self.detector = None
    
    def _extract_wrist_y(self, results) -> Optional[float]:
        """MediaPipe 결과에서 오른쪽 손목 Y좌표 추출"""
        import mediapipe as mp_lib
        if results and results.pose_landmarks:
            wrist = results.pose_landmarks.landmark[mp_lib.solutions.pose.PoseLandmark.RIGHT_WRIST]
            return wrist.y
        return None
    
    def _is_ready_pose(self, results) -> bool:
        """오른쪽 손목이 코보다 위에 있는지 확인"""
        import mediapipe as mp_lib
        if results and results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_lib.solutions.pose.PoseLandmark.NOSE]
            wrist = landmarks[mp_lib.solutions.pose.PoseLandmark.RIGHT_WRIST]
            return wrist.y < (nose.y - 0.05)
        return False
    
    def inference_loop(self):
        """추론 루프"""
        self.logger.info(f"포즈 추론 프로세스 시작 (카메라 {self.camera_index})")
        self.running.value = True
        
        try:
            while self.running.value:
                # 입력 버퍼에서 프레임 읽기
                result = self.input_buffer.read_frame()
                
                if result is None:
                    time.sleep(0.001)  # 짧은 대기
                    continue
                
                frame, timestamp_ns, frame_number = result
                
                # 포즈 추론 수행
                landmarks_data = None
                if self.use_pose_estimation and self.detector is not None:
                    results = self.detector.process(frame)
                    wrist_y = self._extract_wrist_y(results)
                    ready_pose = self._is_ready_pose(results)
                    
                    # 랜드마크 정보를 직렬화 가능한 형태로 변환
                    if results and results.pose_landmarks:
                        landmarks_data = []
                        for landmark in results.pose_landmarks.landmark:
                            landmarks_data.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z,
                                'visibility': landmark.visibility
                            })
                else:
                    wrist_y = None
                    ready_pose = False
                    results = None
                
                # 결과를 큐에 전달
                pose_result = PoseResult(
                    wrist_y=wrist_y,
                    ready_pose=ready_pose,
                    timestamp_ns=timestamp_ns,
                    frame_number=frame_number,
                    landmarks=landmarks_data
                )
                
                # 큐가 가득 찬 경우 오래된 결과 제거 (Non-blocking)
                try:
                    self.output_queue.put(pose_result, block=False)
                except:
                    # 큐가 가득 찬 경우 오래된 항목 제거
                    try:
                        _ = self.output_queue.get_nowait()
                        self.output_queue.put(pose_result, block=False)
                    except:
                        pass  # 큐가 비어있거나 여전히 가득 찬 경우 무시
        
        except KeyboardInterrupt:
            self.logger.info(f"포즈 추론 프로세스 중단 요청 (카메라 {self.camera_index})")
        except Exception as e:
            self.logger.error(f"포즈 추론 프로세스 오류 (카메라 {self.camera_index}): {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def start(self):
        """추론 프로세스 시작"""
        self.running.value = True
        self.inference_loop()
    
    def stop(self):
        """추론 프로세스 중지"""
        self.running.value = False
    
    def cleanup(self):
        """정리 작업"""
        self.running.value = False
        if self.detector is not None:
            self.detector.close()
        self.logger.info(f"포즈 추론 프로세스 정리 완료 (카메라 {self.camera_index})")


def pose_inference_process(
    camera_index: int,
    input_buffer: SharedRingBuffer,
    output_queue: mp.Queue,
    config_path: str = "config.yaml"
):
    """
    포즈 추론 프로세스 함수
    multiprocessing.Process에서 실행되는 함수
    
    Args:
        camera_index: 카메라 인덱스
        input_buffer: 입력 프레임 버퍼
        output_queue: 추론 결과 출력 큐
        config_path: 설정 파일 경로
    """
    try:
        # 공유 메모리 매핑 재설정 (Windows multiprocessing spawn 이슈 대응)
        try:
            input_buffer.ensure_mapping()
        except Exception as e:
            print(f"Pose Inference {camera_index} shared buffer mapping failed: {e}")
            return
        
        # 설정 로드
        try:
            config = load_config(config_path)
        except Exception:
            config = {}
        
        # 추론 프로세스 생성
        inference = PoseInferenceProcess(
            camera_index=camera_index,
            input_buffer=input_buffer,
            output_queue=output_queue,
            config=config,
            config_path=config_path
        )
        
        # 추론 루프 실행
        inference.start()
        
    except Exception as e:
        print(f"포즈 추론 프로세스 {camera_index} 오류: {e}")
        import traceback
        traceback.print_exc()

