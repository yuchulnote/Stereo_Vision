# -*- coding: utf-8 -*-
"""
카메라 Producer 프로세스
별도 프로세스에서 카메라를 캡처하여 공유 메모리에 저장
"""

import cv2
import numpy as np
import multiprocessing as mp
import time
import atexit
import sys
from pathlib import Path
from typing import Optional, Dict

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_buffer import SharedRingBuffer
from utils.camera_config import load_config, apply_camera_settings
from utils.logger import get_logger


class CameraProducer:
    """카메라 Producer 클래스"""
    
    def __init__(
        self,
        camera_index: int,
        buffer: SharedRingBuffer,
        config: Optional[Dict] = None,
        config_path: str = "config.yaml",
        reconnect_interval: float = 1.0
    ):
        """
        Args:
            camera_index: 카메라 인덱스
            buffer: 공유 메모리 버퍼
            config: 카메라 설정 딕셔너리
            config_path: 설정 파일 경로
            reconnect_interval: 재연결 시도 간격 (초)
        """
        self.camera_index = camera_index
        self.buffer = buffer
        self.config = config
        self.config_path = config_path
        self.reconnect_interval = reconnect_interval
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = mp.Value('b', False)
        self.frame_number = mp.Value('i', 0)
        
        # 로거 초기화 (프로세스에서 실행되므로 별도 초기화 필요)
        try:
            # config에서 로거 설정 로드
            full_config = load_config(config_path) if config_path else {}
            logger_config = full_config.get('logging', {})
            self.logger = get_logger(logger_config)
        except Exception:
            # 로거 초기화 실패 시 기본 로거 사용
            import logging
            self.logger = logging.getLogger(f'CameraProducer-{camera_index}')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # 종료 핸들러 등록
        atexit.register(self.cleanup)
    
    def connect_camera(self) -> bool:
        """카메라 연결"""
        try:
            # 기존 연결 해제
            if self.cap is not None:
                self.cap.release()
            
            # 카메라 열기
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                self.logger.error(f"카메라 {self.camera_index}를 열 수 없습니다.")
                return False
            
            # 설정 적용
            if self.config is None:
                try:
                    full_config = load_config(self.config_path)
                    camera_key = f"camera_{self.camera_index}"
                    if camera_key in full_config.get('cameras', {}):
                        self.config = full_config['cameras'][camera_key]
                    else:
                        self.config = {}
                except Exception:
                    self.config = {}
            
            # 카메라 설정 적용 (버퍼 크기 포함)
            apply_camera_settings(self.cap, self.config)
            
            # 버퍼 크기 최소화 확인 (Step 18: 셔터 랙 최소화)
            # config에 buffer_size가 없으면 기본값 1 사용
            if self.config.get('buffer_size') is None:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 버퍼에 맞는 해상도로 강제 설정
            # 버퍼의 해상도와 일치하도록 설정
            buffer_width = self.buffer.width
            buffer_height = self.buffer.height
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, buffer_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, buffer_height)
            
            # 설정 적용 대기
            time.sleep(0.1)
            
            # 실제 해상도 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 버퍼 크기와 일치하는지 확인
            if (actual_width, actual_height) != (buffer_width, buffer_height):
                self.logger.warning(
                    f"카메라 {self.camera_index} 해상도 불일치: "
                    f"요청: {buffer_width}x{buffer_height}, "
                    f"실제: {actual_width}x{actual_height}"
                )
                # 프레임 리사이즈 필요 시 경고
                if actual_width > 0 and actual_height > 0:
                    self.logger.warning(
                        f"프레임이 리사이즈될 수 있습니다. "
                        f"버퍼 크기: {buffer_width}x{buffer_height}"
                    )
            else:
                self.logger.info(
                    f"카메라 {self.camera_index} 연결 성공: "
                    f"{actual_width}x{actual_height} (버퍼와 일치)"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"카메라 {self.camera_index} 연결 중 오류: {e}")
            return False
    
    def capture_loop(self):
        """캡처 루프 (프로세스에서 실행)"""
        self.logger.info(f"카메라 {self.camera_index} Producer 시작")
        
        # 카메라 연결
        if not self.connect_camera():
            self.logger.error(f"카메라 {self.camera_index} 초기 연결 실패")
            return
        
        self.running.value = True
        
        # 첫 프레임 캡처 시도 (연결 확인)
        ret, test_frame = self.cap.read()
        if ret:
            self.logger.info(
                f"카메라 {self.camera_index} 첫 프레임 캡처 성공: "
                f"{test_frame.shape}"
            )
        else:
            self.logger.warning(f"카메라 {self.camera_index} 첫 프레임 캡처 실패")
        
        try:
            while self.running.value:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning(f"카메라 {self.camera_index}에서 프레임 읽기 실패")
                    
                    # 재연결 시도 (Step 23)
                    self.cap.release()
                    time.sleep(self.reconnect_interval)
                    
                    if not self.connect_camera():
                        self.logger.error(f"카메라 {self.camera_index} 재연결 실패")
                        time.sleep(self.reconnect_interval)
                        continue
                    
                    continue
                
                # 프레임 크기 확인 및 리사이즈 (버퍼 크기와 일치하도록)
                buffer_width = self.buffer.width
                buffer_height = self.buffer.height
                
                # 프레임 크기 확인
                expected_shape = (buffer_height, buffer_width, self.buffer.channels)
                if frame.shape != expected_shape:
                    # 프레임 리사이즈
                    old_shape = frame.shape
                    frame = cv2.resize(frame, (buffer_width, buffer_height))
                    if not hasattr(self, '_resize_logged'):
                        self.logger.warning(
                            f"카메라 {self.camera_index} 프레임 리사이즈: "
                            f"{old_shape} -> {frame.shape}"
                        )
                        self._resize_logged = True
                
                # 최종 크기 확인
                if frame.shape != expected_shape:
                    self.logger.error(
                        f"카메라 {self.camera_index} 프레임 크기 불일치: "
                        f"프레임={frame.shape}, 예상={expected_shape}"
                    )
                    continue  # 이 프레임 건너뛰기
                
                # 타임스탬프 (나노초)
                timestamp_ns = time.time_ns()
                
                # 프레임 번호
                with self.frame_number.get_lock():
                    frame_num = self.frame_number.value
                    self.frame_number.value += 1
                
                # 공유 메모리에 쓰기
                # 디버깅: 쓰기 전 프레임 평균값 확인
                if not hasattr(self, '_frame_check_count'):
                    self._frame_check_count = 0
                
                self._frame_check_count += 1
                if self._frame_check_count <= 20:  # 처음 20프레임 확인
                    mean_val = np.mean(frame)
                    if mean_val < 1.0:
                        self.logger.warning(f"Producer 캡처 프레임 어두움 (평균값: {mean_val:.1f})")
                    else:
                        self.logger.info(f"Producer 캡처 프레임 정상 (평균값: {mean_val:.1f})")
                
                success = self.buffer.write_frame(frame, timestamp_ns, frame_num)
                
                # 디버깅: 처음 몇 번만 로그
                if not hasattr(self, '_write_debug_count'):
                    self._write_debug_count = 0
                
                self._write_debug_count += 1
                if self._write_debug_count <= 20:  # 처음 20번은 항상 로그
                    if success:
                        self.logger.info(
                            f"카메라 {self.camera_index} 버퍼 쓰기 성공 "
                            f"(프레임 번호: {frame_num}, 시도: {self._write_debug_count})"
                        )
                    else:
                        # 이 경우는 프레임 크기 불일치 등의 오류
                        self.logger.error(
                            f"카메라 {self.camera_index} 버퍼 쓰기 실패 "
                            f"(프레임 크기: {frame.shape}, 버퍼 크기: {self.buffer.width}x{self.buffer.height}, "
                            f"프레임 번호: {frame_num})"
                        )
                elif self._write_debug_count == 21:
                    # 이후에는 주기적으로만 로그
                    self.logger.info(f"카메라 {self.camera_index} 버퍼 쓰기 계속 진행 중... (로그 빈도 감소)")
        
        except KeyboardInterrupt:
            self.logger.info(f"카메라 {self.camera_index} Producer 중단 요청")
        except Exception as e:
            self.logger.error(f"카메라 {self.camera_index} Producer 오류: {e}")
        finally:
            self.cleanup()
    
    def start(self):
        """Producer 시작"""
        self.running.value = True
        self.capture_loop()
    
    def stop(self):
        """Producer 중지"""
        self.running.value = False
        if self.cap is not None:
            self.cap.release()
    
    def cleanup(self):
        """정리 작업"""
        self.running.value = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.logger.info(f"카메라 {self.camera_index} Producer 정리 완료")


def camera_producer_process(
    camera_index: int,
    buffer: SharedRingBuffer,
    config_path: str = "config.yaml",
    reconnect_interval: float = 1.0
):
    """
    카메라 Producer 프로세스 함수
    multiprocessing.Process에서 실행되는 함수
    
    Args:
        camera_index: 카메라 인덱스
        buffer: 공유 메모리 버퍼 객체 (SharedRingBuffer)
        config_path: 설정 파일 경로
        reconnect_interval: 재연결 간격
    """
    try:
        # 공유 메모리 매핑 재설정 (Windows multiprocessing spawn 이슈 대응)
        try:
            buffer.ensure_mapping()
        except Exception as e:
            print(f"Camera {camera_index} shared buffer mapping failed: {e}")
            # 매핑 실패 시 실행 중단
            return

        # Producer 생성
        producer = CameraProducer(
            camera_index=camera_index,
            buffer=buffer,
            config_path=config_path,
            reconnect_interval=reconnect_interval
        )
        
        # 캡처 루프 실행
        producer.start()
        
    except Exception as e:
        print(f"카메라 {camera_index} Producer 프로세스 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        # buffer는 외부에서 주입되었으므로 여기서 닫지 않음 (메인 프로세스 관리)
        pass

