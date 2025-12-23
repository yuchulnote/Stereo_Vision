# -*- coding: utf-8 -*-
"""
카메라 설정 적용 유틸리티
config.yaml의 설정값을 카메라에 적용합니다.
"""

import cv2
import yaml
from pathlib import Path
from typing import Dict, Optional


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def apply_camera_settings(cap: cv2.VideoCapture, camera_config: Dict) -> bool:
    """
    카메라에 설정값 적용
    
    Args:
        cap: cv2.VideoCapture 객체
        camera_config: 카메라 설정 딕셔너리
        
    Returns:
        성공 여부
    """
    if not cap.isOpened():
        return False
    
    try:
        # 해상도 설정
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # FPS 설정
        fps = camera_config.get('fps', 30)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 자동 노출 비활성화 (플리커 방지)
        auto_exposure = camera_config.get('auto_exposure', False)
        if not auto_exposure:
            # 자동 노출 모드 비활성화
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = 수동 모드
            exposure = camera_config.get('exposure', -6)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = 자동 모드
        
        # 화이트 밸런스 설정
        auto_white_balance = camera_config.get('auto_white_balance', False)
        if not auto_white_balance:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # 자동 화이트 밸런스 비활성화
            white_balance = camera_config.get('white_balance', 5000)
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, white_balance)
        else:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # 자동 화이트 밸런스 활성화
        
        # 포커스 설정
        auto_focus = camera_config.get('auto_focus', False)
        if not auto_focus:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 자동 포커스 비활성화
            focus = camera_config.get('focus', 50)
            cap.set(cv2.CAP_PROP_FOCUS, focus)
        else:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 자동 포커스 활성화
        
        # 기타 설정
        brightness = camera_config.get('brightness', 128)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        
        contrast = camera_config.get('contrast', 128)
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
        
        saturation = camera_config.get('saturation', 128)
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
        
        sharpness = camera_config.get('sharpness', 128)
        cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
        
        # 버퍼 크기 설정 (Step 18: 셔터 랙 최소화)
        buffer_size = camera_config.get('buffer_size', 1)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        return True
        
    except Exception as e:
        print(f"카메라 설정 적용 중 오류 발생: {e}")
        return False


def open_camera_with_config(camera_index: int, config_path: str = "config.yaml") -> Optional[cv2.VideoCapture]:
    """
    설정 파일을 사용하여 카메라 열기
    
    Args:
        camera_index: 카메라 인덱스
        config_path: 설정 파일 경로
        
    Returns:
        cv2.VideoCapture 객체 또는 None
    """
    try:
        config = load_config(config_path)
        
        # 카메라 설정 찾기
        camera_key = f"camera_{camera_index}"
        if camera_key not in config.get('cameras', {}):
            # 기본 설정 사용
            camera_config = {
                'width': 640,
                'height': 480,
                'fps': 30,
                'auto_exposure': False,
                'exposure': -6,
            }
        else:
            camera_config = config['cameras'][camera_key]
        
        # 카메라 열기
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            return None
        
        # 설정 적용
        if apply_camera_settings(cap, camera_config):
            return cap
        else:
            cap.release()
            return None
            
    except Exception as e:
        print(f"카메라 열기 중 오류 발생: {e}")
        return None

