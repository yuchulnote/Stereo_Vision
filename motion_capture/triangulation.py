# -*- coding: utf-8 -*-
"""
삼각 측량 (Triangulation) 알고리즘
2개의 2D 좌표와 투영 행렬을 이용해 3D 좌표 복원
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.linalg import svd


def triangulate_points_dlt(
    point1: np.ndarray,
    point2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray
) -> np.ndarray:
    """
    DLT (Direct Linear Transformation) 기반 삼각 측량
    
    Args:
        point1: 카메라 1의 2D 좌표 (u, v)
        point2: 카메라 2의 2D 좌표 (u, v)
        P1: 카메라 1의 투영 행렬 (3x4)
        P2: 카메라 2의 투영 행렬 (3x4)
        
    Returns:
        3D 좌표 (X, Y, Z, W) - 동차 좌표
    """
    # DLT 행렬 구성
    A = np.zeros((4, 4))
    
    # 카메라 1 제약 조건
    A[0] = point1[0] * P1[2] - P1[0]
    A[1] = point1[1] * P1[2] - P1[1]
    
    # 카메라 2 제약 조건
    A[2] = point2[0] * P2[2] - P2[0]
    A[3] = point2[1] * P2[2] - P2[1]
    
    # SVD를 이용한 해 구하기
    U, s, Vt = svd(A)
    X = Vt[-1, :]  # 가장 작은 특이값에 해당하는 벡터
    
    return X


def triangulate_points_midpoint(
    point1: np.ndarray,
    point2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    R1: np.ndarray,
    T1: np.ndarray,
    R2: np.ndarray,
    T2: np.ndarray
) -> np.ndarray:
    """
    Mid-point 근사 기반 삼각 측량
    두 카메라의 Ray가 정확히 만나지 않는 문제를 해결
    
    Args:
        point1: 카메라 1의 2D 좌표 (u, v)
        point2: 카메라 2의 2D 좌표 (u, v)
        P1: 카메라 1의 투영 행렬 (3x4)
        P2: 카메라 2의 투영 행렬 (3x4)
        K1: 카메라 1의 내부 파라미터 행렬 (3x3)
        K2: 카메라 2의 내부 파라미터 행렬 (3x3)
        R1: 카메라 1의 회전 행렬 (3x3)
        T1: 카메라 1의 변위 벡터 (3x1)
        R2: 카메라 2의 회전 행렬 (3x3)
        T2: 카메라 2의 변위 벡터 (3x1)
        
    Returns:
        3D 좌표 (X, Y, Z)
    """
    # 카메라 중심 계산
    # T 벡터가 (3, 1) 형태일 수 있으므로 flatten()으로 1차원 배열로 변환하여 브로드캐스팅 문제(3,3 생성) 방지
    T1_flat = T1.flatten()
    T2_flat = T2.flatten()
    
    C1 = -R1.T @ T1_flat  # 카메라 1의 중심 (월드 좌표) -> (3,)
    C2 = -R2.T @ T2_flat  # 카메라 2의 중심 (월드 좌표) -> (3,)
    
    # 정규화된 이미지 좌표로 변환
    p1_homogeneous = np.array([point1[0], point1[1], 1.0])
    p2_homogeneous = np.array([point2[0], point2[1], 1.0])
    
    # 역 투영 (Ray 방향 계산)
    K1_inv = np.linalg.inv(K1)
    K2_inv = np.linalg.inv(K2)
    
    ray1_camera = K1_inv @ p1_homogeneous
    ray2_camera = K2_inv @ p2_homogeneous
    
    # 월드 좌표계로 변환
    ray1_world = R1.T @ ray1_camera
    ray2_world = R2.T @ ray2_camera
    
    # 정규화
    ray1_world = ray1_world / np.linalg.norm(ray1_world)
    ray2_world = ray2_world / np.linalg.norm(ray2_world)
    
    # 두 Ray의 가장 가까운 점 계산
    # 두 선분 사이의 최단 거리 점을 찾는 문제
    w = C1 - C2
    a = np.dot(ray1_world, ray1_world)
    b = np.dot(ray1_world, ray2_world)
    c = np.dot(ray2_world, ray2_world)
    d = np.dot(ray1_world, w)
    e = np.dot(ray2_world, w)
    
    denom = a * c - b * b
    if abs(denom) < 1e-6:
        # 평행한 경우 DLT 사용
        return triangulate_points_dlt(point1, point2, P1, P2)[:3]
    
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    # 두 Ray 위의 점
    P1_3d = C1 + t1 * ray1_world
    P2_3d = C2 + t2 * ray2_world
    
    # Mid-point (두 점의 중점)
    P_3d = (P1_3d + P2_3d) / 2.0
    
    return P_3d


def triangulate_landmarks(
    landmarks_2d_0: list,
    landmarks_2d_1: list,
    P1: np.ndarray,
    P2: np.ndarray,
    K1: Optional[np.ndarray] = None,
    K2: Optional[np.ndarray] = None,
    R1: Optional[np.ndarray] = None,
    T1: Optional[np.ndarray] = None,
    R2: Optional[np.ndarray] = None,
    T2: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.6,
    use_midpoint: bool = True,
    image_size: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    여러 랜드마크에 대해 삼각 측량 수행
    
    Args:
        landmarks_2d_0: 카메라 0의 2D 랜드마크 리스트
        landmarks_2d_1: 카메라 1의 2D 랜드마크 리스트
        P1: 카메라 1의 투영 행렬 (3x4)
        P2: 카메라 2의 투영 행렬 (3x4)
        K1: 카메라 1의 내부 파라미터 (선택적, midpoint 사용 시 필요)
        K2: 카메라 2의 내부 파라미터 (선택적, midpoint 사용 시 필요)
        R1, T1, R2, T2: 카메라 외부 파라미터 (선택적, midpoint 사용 시 필요)
        confidence_threshold: 신뢰도 임계값
        use_midpoint: Mid-point 방법 사용 여부
        
    Returns:
        (landmarks_3d, valid_mask)
        landmarks_3d: 3D 랜드마크 배열 (N, 3)
        valid_mask: 유효한 랜드마크 마스크 (N,)
    """
    assert len(landmarks_2d_0) == len(landmarks_2d_1), "랜드마크 개수가 일치해야 합니다"
    
    n_landmarks = len(landmarks_2d_0)
    landmarks_3d = np.zeros((n_landmarks, 3))
    valid_mask = np.zeros(n_landmarks, dtype=bool)
    
    for i in range(n_landmarks):
        lm0 = landmarks_2d_0[i]
        lm1 = landmarks_2d_1[i]
        
        # 신뢰도 체크
        vis0 = lm0.get('visibility', 0.0)
        vis1 = lm1.get('visibility', 0.0)
        
        if vis0 < confidence_threshold or vis1 < confidence_threshold:
            continue
        
        # 2D 좌표 추출 (MediaPipe는 정규화된 좌표 0~1 사용)
        # 픽셀 좌표로 변환하려면 이미지 크기가 필요하지만,
        # 투영 행렬에 이미 포함되어 있으므로 정규화된 좌표를 그대로 사용
        # 단, OpenCV는 픽셀 좌표를 기대하므로 이미지 크기를 가정하거나
        # 투영 행렬에서 추출해야 함
        # 여기서는 정규화된 좌표를 그대로 사용 (투영 행렬이 정규화된 좌표를 처리하도록)
        point1 = np.array([lm0['x'], lm0['y']])
        point2 = np.array([lm1['x'], lm1['y']])
        
        # 삼각 측량 수행
        # MediaPipe는 정규화된 좌표(0~1)를 사용하므로 픽셀 좌표로 변환 필요
        # 이미지 크기는 투영 행렬에서 추정하거나 별도로 전달받아야 함
        # 여기서는 정규화된 좌표를 픽셀 좌표로 변환 (이미지 크기 가정: 1920x1080)
        # 이미지 크기 가져오기
        if image_size is not None:
            image_width, image_height = image_size
        else:
            image_width = 1920  # 기본값
            image_height = 1080
        
        # 정규화된 좌표를 픽셀 좌표로 변환
        pixel_point1 = np.array([point1[0] * image_width, point1[1] * image_height])
        pixel_point2 = np.array([point2[0] * image_width, point2[1] * image_height])
        
        if use_midpoint and K1 is not None and K2 is not None:
            try:
                point_3d = triangulate_points_midpoint(
                    pixel_point1, pixel_point2, P1, P2, K1, K2, R1, T1, R2, T2
                )
            except:
                # 실패 시 DLT 사용
                point_3d_homogeneous = triangulate_points_dlt(pixel_point1, pixel_point2, P1, P2)
                point_3d = point_3d_homogeneous[:3] / point_3d_homogeneous[3]
        else:
            # OpenCV의 triangulatePoints 사용 (DLT 기반)
            points_2d_1 = pixel_point1.reshape(2, 1).astype(np.float32)
            points_2d_2 = pixel_point2.reshape(2, 1).astype(np.float32)
            
            point_3d_homogeneous = cv2.triangulatePoints(P1, P2, points_2d_1, points_2d_2)
            point_3d = point_3d_homogeneous[:3, 0] / point_3d_homogeneous[3, 0]
        
        landmarks_3d[i] = point_3d
        valid_mask[i] = True
    
    return landmarks_3d, valid_mask

