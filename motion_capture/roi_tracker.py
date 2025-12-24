# -*- coding: utf-8 -*-
"""
ROI (Region of Interest) Tracking
이전 프레임의 사람 위치를 기반으로 다음 프레임의 탐색 영역을 좁혀 추론 속도 향상
"""

import numpy as np
from typing import Optional, Tuple, List
import cv2


class ROITracker:
    """ROI 추적 클래스"""
    
    def __init__(
        self,
        margin_ratio: float = 0.2,
        decay_factor: float = 0.9,
        min_roi_size: Tuple[int, int] = (100, 100)
    ):
        """
        Args:
            margin_ratio: ROI 주변 마진 비율 (0.2 = 20% 여유)
            decay_factor: ROI 크기 감소 계수 (0.9 = 매 프레임마다 10% 감소)
            min_roi_size: 최소 ROI 크기 (width, height)
        """
        self.margin_ratio = margin_ratio
        self.decay_factor = decay_factor
        self.min_roi_size = min_roi_size
        
        # 현재 ROI 상태
        self.current_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
        self.confidence: float = 0.0  # ROI 신뢰도 (0.0 ~ 1.0)
        self.frame_count_without_detection = 0
    
    def update(self, landmarks: Optional[List[dict]], image_shape: Tuple[int, ...]) -> Optional[Tuple[int, int, int, int]]:
        """
        랜드마크를 기반으로 ROI 업데이트
        
        Args:
            landmarks: MediaPipe 랜드마크 리스트 (None이면 감지 실패)
            image_shape: 이미지 크기 (height, width) 또는 (height, width, channels)
            
        Returns:
            ROI (x, y, width, height) 또는 None
        """
        # image_shape가 (height, width, channels) 형태일 수 있으므로 처리
        if len(image_shape) >= 2:
            img_height, img_width = image_shape[0], image_shape[1]
        else:
            raise ValueError(f"image_shape는 최소 2개의 차원을 가져야 합니다: {image_shape}")
        
        if landmarks is None or len(landmarks) == 0:
            # 감지 실패 시 ROI 감소
            self.frame_count_without_detection += 1
            
            if self.current_roi is not None:
                # 기존 ROI를 점진적으로 확대 (전체 이미지로 복귀)
                x, y, w, h = self.current_roi
                center_x = x + w // 2
                center_y = y + h // 2
                
                # ROI를 점진적으로 확대
                new_w = min(int(w * (1.0 + (1.0 - self.decay_factor))), img_width)
                new_h = min(int(h * (1.0 + (1.0 - self.decay_factor))), img_height)
                
                # 중심 유지하며 확대
                new_x = max(0, center_x - new_w // 2)
                new_y = max(0, center_y - new_h // 2)
                new_x = min(new_x, img_width - new_w)
                new_y = min(new_y, img_height - new_h)
                
                self.current_roi = (new_x, new_y, new_w, new_h)
                self.confidence *= self.decay_factor
                
                # 너무 낮은 신뢰도면 ROI 초기화
                if self.confidence < 0.1 or self.frame_count_without_detection > 30:
                    self.current_roi = None
                    self.confidence = 0.0
                    return None
            else:
                return None
        
        # 랜드마크가 있으면 ROI 계산
        self.frame_count_without_detection = 0
        
        # landmarks가 None이거나 비어있으면 처리하지 않음
        if not landmarks:
            return self.current_roi
        
        # 모든 랜드마크의 x, y 좌표 추출
        visible_landmarks = [
            (lm['x'] * img_width, lm['y'] * img_height)
            for lm in landmarks
            if lm is not None and lm.get('visibility', 0) > 0.5
        ]
        
        if len(visible_landmarks) < 5:  # 최소 5개 랜드마크 필요
            return self.current_roi
        
        # 바운딩 박스 계산
        xs = [x for x, y in visible_landmarks]
        ys = [y for x, y in visible_landmarks]
        
        min_x = max(0, int(min(xs)))
        max_x = min(img_width, int(max(xs)))
        min_y = max(0, int(min(ys)))
        max_y = min(img_height, int(max(ys)))
        
        width = max_x - min_x
        height = max_y - min_y
        
        # 마진 추가
        margin_x = int(width * self.margin_ratio)
        margin_y = int(height * self.margin_ratio)
        
        x = max(0, min_x - margin_x)
        y = max(0, min_y - margin_y)
        w = min(img_width - x, width + 2 * margin_x)
        h = min(img_height - y, height + 2 * margin_y)
        
        # 최소 크기 보장
        if w < self.min_roi_size[0]:
            center_x = x + w // 2
            w = self.min_roi_size[0]
            x = max(0, center_x - w // 2)
            x = min(x, img_width - w)
        
        if h < self.min_roi_size[1]:
            center_y = y + h // 2
            h = self.min_roi_size[1]
            y = max(0, center_y - h // 2)
            y = min(y, img_height - h)
        
        new_roi = (x, y, w, h)
        
        # 기존 ROI와의 일치도 계산 (신뢰도)
        if self.current_roi is not None:
            old_x, old_y, old_w, old_h = self.current_roi
            old_center = (old_x + old_w // 2, old_y + old_h // 2)
            new_center = (x + w // 2, y + h // 2)
            
            # 중심 거리 기반 신뢰도
            distance = np.sqrt((old_center[0] - new_center[0])**2 + (old_center[1] - new_center[1])**2)
            max_distance = np.sqrt(img_width**2 + img_height**2)
            position_confidence = 1.0 - min(1.0, distance / (max_distance * 0.1))
            
            # 크기 변화 기반 신뢰도
            size_ratio = min(old_w, old_h) / max(old_w, old_h) if max(old_w, old_h) > 0 else 0
            new_size_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
            size_confidence = 1.0 - abs(size_ratio - new_size_ratio)
            
            self.confidence = (position_confidence + size_confidence) / 2.0
        else:
            self.confidence = 0.7  # 초기 신뢰도
        
        self.current_roi = new_roi
        return new_roi
    
    def get_roi(self) -> Optional[Tuple[int, int, int, int]]:
        """현재 ROI 반환"""
        return self.current_roi
    
    def reset(self):
        """ROI 초기화"""
        self.current_roi = None
        self.confidence = 0.0
        self.frame_count_without_detection = 0

