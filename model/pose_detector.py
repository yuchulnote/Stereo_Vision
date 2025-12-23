# -*- coding: utf-8 -*-
"""
MediaPipe Pose Detector
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Tuple, List
import os

# [디버깅용 코드] 경로 출력
print("="*50)
print(f"현재 로드된 mediapipe 경로: {os.path.dirname(mp.__file__) if hasattr(mp, '__file__') else '경로 없음 (Namespace Package)'}")
print("="*50)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class PoseDetector:
    """MediaPipe를 사용한 포즈 추정 클래스"""
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_cuda: bool = False
    ):
        """
        Args:
            static_image_mode: 정지 이미지 모드 여부
            model_complexity: 모델 복잡도 (0: Lite, 1: Full, 2: Heavy)
            smooth_landmarks: 랜드마크 스무딩 여부
            enable_segmentation: 세그멘테이션 활성화 여부
            smooth_segmentation: 세그멘테이션 스무딩 여부
            min_detection_confidence: 감지 최소 신뢰도
            min_tracking_confidence: 추적 최소 신뢰도
            use_cuda: CUDA 사용 여부 (MediaPipe Python은 현재 GPU를 직접 지원하지 않으므로 경고 출력)
        """
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        if use_cuda:
            # 경고 메시지는 한 번만 출력하거나 로거를 통해 출력하는 것이 좋음
            pass
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    def process(self, image: np.ndarray) -> object:
        """
        이미지에서 포즈 추정
        
        Args:
            image: 입력 이미지 (BGR)
            
        Returns:
            results: MediaPipe Pose 결과 객체
        """
        # MediaPipe는 RGB 이미지를 사용하므로 변환 필요
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 성능 향상을 위해 이미지 쓰기 방지 설정
        image_rgb.flags.writeable = False
        
        # 추론 수행
        results = self.pose.process(image_rgb)
        
        return results
    
    def draw_landmarks(self, image: np.ndarray, results: object):
        """
        이미지에 랜드마크 그리기
        
        Args:
            image: 그릴 대상 이미지 (BGR)
            results: process() 메서드의 결과
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
    def close(self):
        """자원 해제"""
        try:
            if hasattr(self, 'pose') and self.pose is not None:
                self.pose.close()
        except Exception as e:
            # 이미 닫혔거나 해제된 경우 무시
            pass
        finally:
            self.pose = None
