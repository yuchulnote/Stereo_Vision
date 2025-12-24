# -*- coding: utf-8 -*-
"""
3D 모션캡쳐 메인 모듈
2D 포즈 추정 결과를 3D로 복원하고 필터링
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from motion_capture.triangulation import triangulate_landmarks
from motion_capture.one_euro_filter import OneEuroFilter3D
from utils.logger import get_logger


class MotionCapture3D:
    """3D 모션캡쳐 클래스"""
    
    def __init__(
        self,
        calibration_data: Dict,
        confidence_threshold: float = 0.6,
        use_midpoint: bool = True,
        filter_enabled: bool = True,
        filter_min_cutoff: float = 1.0,
        filter_beta: float = 0.5,
        filter_d_cutoff: float = 1.0,
        filter_freq: float = 30.0
    ):
        """
        Args:
            calibration_data: 캘리브레이션 데이터 (calibration_result.yaml에서 로드)
            confidence_threshold: 신뢰도 임계값
            use_midpoint: Mid-point 방법 사용 여부
            filter_enabled: 필터 활성화 여부
            filter_min_cutoff: 필터 최소 차단 주파수
            filter_beta: 필터 속도 감쇠 계수
            filter_d_cutoff: 필터 속도 차단 주파수
            filter_freq: 필터 샘플링 주파수
        """
        self.calibration_data = calibration_data
        self.confidence_threshold = confidence_threshold
        self.use_midpoint = use_midpoint
        self.filter_enabled = filter_enabled
        
        self.logger = get_logger()
        
        # 투영 행렬 계산
        self.P1, self.P2, self.K1, self.K2, self.R1, self.T1, self.R2, self.T2 = self._compute_projection_matrices()
        
        # 필터 초기화 (각 랜드마크마다)
        self.filters: List[Optional[OneEuroFilter3D]] = []
        self.filter_freq = filter_freq
        
        # MediaPipe Pose 랜드마크 개수 (33개)
        self.n_landmarks = 33
        for _ in range(self.n_landmarks):
            if filter_enabled:
                self.filters.append(OneEuroFilter3D(
                    min_cutoff=filter_min_cutoff,
                    beta=filter_beta,
                    d_cutoff=filter_d_cutoff,
                    freq=filter_freq
                ))
            else:
                self.filters.append(None)
    
    def _compute_projection_matrices(self) -> Tuple:
        """캘리브레이션 데이터로부터 투영 행렬 계산"""
        # 내부 파라미터
        K1 = np.array(self.calibration_data['camera_matrix_left'])
        K2 = np.array(self.calibration_data['camera_matrix_right'])
        
        # 외부 파라미터
        R = np.array(self.calibration_data['R'])  # 상대 회전
        T = np.array(self.calibration_data['T'])  # 상대 변위
        
        # 카메라 1은 기준 (단위 행렬)
        R1 = np.eye(3)
        T1 = np.zeros((3, 1))
        
        # 카메라 2는 상대 변환 적용
        R2 = R
        T2 = T
        
        # 투영 행렬 계산: P = K [R | t]
        P1 = K1 @ np.hstack([R1, T1])
        P2 = K2 @ np.hstack([R2, T2])
        
        return P1, P2, K1, K2, R1, T1, R2, T2
    
    def process(
        self,
        landmarks_2d_0: List[Dict],
        landmarks_2d_1: List[Dict],
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D 랜드마크를 3D로 복원
        
        Args:
            landmarks_2d_0: 카메라 0의 2D 랜드마크
            landmarks_2d_1: 카메라 1의 2D 랜드마크
            timestamp: 타임스탬프
            
        Returns:
            (landmarks_3d, valid_mask)
            landmarks_3d: 3D 랜드마크 배열 (N, 3)
            valid_mask: 유효한 랜드마크 마스크 (N,)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 이미지 크기 가져오기
        image_size = tuple(self.calibration_data.get('image_size', (1920, 1080)))
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        
        # 삼각 측량 수행
        landmarks_3d, valid_mask = triangulate_landmarks(
            landmarks_2d_0,
            landmarks_2d_1,
            self.P1,
            self.P2,
            K1=self.K1,
            K2=self.K2,
            R1=self.R1,
            T1=self.T1,
            R2=self.R2,
            T2=self.T2,
            confidence_threshold=self.confidence_threshold,
            use_midpoint=self.use_midpoint,
            image_size=image_size
        )
        
        # 필터링 적용
        if self.filter_enabled:
            for i in range(len(landmarks_3d)):
                if valid_mask[i] and self.filters[i] is not None:
                    landmarks_3d[i] = self.filters[i].filter(landmarks_3d[i], timestamp)
        
        return landmarks_3d, valid_mask
    
    def reset_filters(self):
        """필터 초기화"""
        for filter_obj in self.filters:
            if filter_obj is not None:
                filter_obj.reset()
    
    def update_filter_freq(self, freq: float):
        """필터 주파수 업데이트"""
        self.filter_freq = freq
        # 필터 재생성은 복잡하므로 주의 필요
        # 필요시 필터를 재생성하는 로직 추가 가능

