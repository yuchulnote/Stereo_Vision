# -*- coding: utf-8 -*-
"""
OneEuroFilter: 속도에 따라 적응적으로 변하는 필터
빠른 움직임은 유지하되 미세한 떨림만 제거
"""

import numpy as np
from typing import Optional
import time


class LowPassFilter:
    """저역 통과 필터"""
    
    def __init__(self, alpha: float):
        """
        Args:
            alpha: 필터 계수 (0~1, 1에 가까울수록 빠른 반응)
        """
        self.alpha = alpha
        self.last_value: Optional[np.ndarray] = None
    
    def filter(self, value: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        """
        필터링 수행
        
        Args:
            value: 입력 값
            alpha: 필터 계수 (None이면 초기값 사용)
        """
        if alpha is None:
            alpha = self.alpha
        
        if self.last_value is None:
            self.last_value = value.copy()
            return value
        
        filtered = alpha * value + (1 - alpha) * self.last_value
        self.last_value = filtered
        return filtered
    
    def reset(self):
        """필터 초기화"""
        self.last_value = None


class OneEuroFilter:
    """OneEuroFilter: 속도 기반 적응 필터"""
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        freq: float = 30.0
    ):
        """
        Args:
            min_cutoff: 최소 차단 주파수 (낮을수록 더 부드러움)
            beta: 속도 감쇠 계수 (높을수록 빠른 움직임에 더 반응)
            d_cutoff: 속도 필터의 차단 주파수
            freq: 샘플링 주파수 (FPS)
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.freq = freq
        
        self.x_filter = LowPassFilter(self._alpha(min_cutoff))
        self.dx_filter = LowPassFilter(self._alpha(d_cutoff))
        
        self.last_time: Optional[float] = None
    
    def _alpha(self, cutoff: float) -> float:
        """차단 주파수로부터 alpha 계산"""
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)
    
    def filter(self, value: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        필터링 수행
        
        Args:
            value: 입력 값 (numpy array)
            timestamp: 타임스탬프 (None이면 현재 시간 사용)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # 속도 계산 (이전 값과의 차이)
        if self.last_time is None:
            self.last_time = timestamp
            dx = np.zeros_like(value)
        else:
            dt = timestamp - self.last_time
            if dt <= 0:
                dt = 1.0 / self.freq
            
            # 이전 필터링된 값 사용
            if self.x_filter.last_value is not None:
                dx = (value - self.x_filter.last_value) / dt
            else:
                dx = np.zeros_like(value)
            
            self.last_time = timestamp
        
        # 속도 필터링
        dx_filtered = self.dx_filter.filter(dx)
        
        # 적응적 차단 주파수 계산
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_filtered)
        
        # 위치 필터링 (적응적 alpha 사용)
        alpha = self._alpha(cutoff)
        filtered_value = self.x_filter.filter(value, alpha)
        
        return filtered_value
    
    def reset(self):
        """필터 초기화"""
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class OneEuroFilter3D:
    """3D 좌표용 OneEuroFilter (각 축별로 독립적으로 필터링)"""
    
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
        freq: float = 30.0
    ):
        """
        Args:
            min_cutoff: 최소 차단 주파수
            beta: 속도 감쇠 계수
            d_cutoff: 속도 필터의 차단 주파수
            freq: 샘플링 주파수
        """
        self.filter_x = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
        self.filter_y = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
        self.filter_z = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)
    
    def filter(self, point_3d: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        """
        3D 점 필터링
        
        Args:
            point_3d: 3D 좌표 (3,) 또는 (N, 3)
            timestamp: 타임스탬프
        """
        if point_3d.ndim == 1:
            # 단일 점
            x = self.filter_x.filter(np.array([point_3d[0]]), timestamp)[0]
            y = self.filter_y.filter(np.array([point_3d[1]]), timestamp)[0]
            z = self.filter_z.filter(np.array([point_3d[2]]), timestamp)[0]
            return np.array([x, y, z])
        else:
            # 여러 점
            filtered = np.zeros_like(point_3d)
            for i in range(point_3d.shape[0]):
                filtered[i] = self.filter(point_3d[i], timestamp)
            return filtered
    
    def reset(self):
        """필터 초기화"""
        self.filter_x.reset()
        self.filter_y.reset()
        self.filter_z.reset()

