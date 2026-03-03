# -*- coding: utf-8 -*-
"""
인체 관절 기반 스테레오 자동 캘리브레이션
체스보드 없이 사람이 걸어다니는 것만으로 F → E → R, T 복원
인체 비율로 절대 스케일 복구
"""

import numpy as np
import cv2
import yaml
from typing import Optional, Dict, List, Tuple


# COCO 17-keypoint 중 캘리브레이션에 사용할 안정적인 10개 관절
# 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow,
# 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
CALIBRATION_JOINT_INDICES = [5, 6, 7, 8, 11, 12, 13, 14, 15, 16]

# 인체 세그먼트 길이 (mm) — 성인 평균
# (joint_a_idx, joint_b_idx, true_length_mm)
# 모든 인덱스는 CALIBRATION_JOINT_INDICES에 포함된 관절만 사용
BODY_SEGMENTS = [
    (5, 6, 400),     # 어깨 너비 ~40cm
    (5, 7, 280),     # 좌 상완 ~28cm
    (6, 8, 280),     # 우 상완 ~28cm
    (11, 12, 280),   # 골반 너비 ~28cm
    (11, 13, 430),   # 좌 대퇴 ~43cm
    (12, 14, 430),   # 우 대퇴 ~43cm
    (13, 15, 400),   # 좌 정강이 ~40cm
    (14, 16, 400),   # 우 정강이 ~40cm
    (5, 11, 500),    # 좌 몸통 ~50cm
    (6, 12, 500),    # 우 몸통 ~50cm
]

# 캘리브레이션 최소 요구사항
MIN_POINTS = 30
MIN_COVERAGE = 6       # 16칸 중 6칸 이상
GRID_SIZE = 4


class AutoCalibrator:
    """인체 관절 기반 스테레오 자동 캘리브레이션

    RTMPose COCO-17 형식의 keypoint 대응점을 수집하고,
    RANSAC으로 F → E → R, T를 복원한 뒤
    인체 세그먼트 비율로 절대 스케일을 복구한다.

    출력 dict는 MotionCapture3D(mocap_3d.py)에 그대로 전달 가능.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        intrinsic_left: Optional[np.ndarray] = None,
        intrinsic_right: Optional[np.ndarray] = None,
    ):
        """
        Args:
            image_size: (width, height)
            intrinsic_left: 3x3 카메라 내부 행렬 (None이면 f=max(w,h)로 추정)
            intrinsic_right: 3x3 카메라 내부 행렬
        """
        self.image_size = image_size
        w, h = image_size

        if intrinsic_left is not None:
            self.K1 = intrinsic_left.astype(np.float64)
        else:
            f = float(max(w, h))
            self.K1 = np.array([[f, 0, w / 2],
                                [0, f, h / 2],
                                [0, 0, 1]], dtype=np.float64)

        if intrinsic_right is not None:
            self.K2 = intrinsic_right.astype(np.float64)
        else:
            f = float(max(w, h))
            self.K2 = np.array([[f, 0, w / 2],
                                [0, f, h / 2],
                                [0, 0, 1]], dtype=np.float64)

        self.reset()

    # ------------------------------------------------------------------
    # 데이터 수집
    # ------------------------------------------------------------------

    def reset(self):
        """재캘리브레이션용 초기화"""
        self.points_cam0: List[np.ndarray] = []
        self.points_cam1: List[np.ndarray] = []
        self.point_joint_ids: List[int] = []
        self.point_frame_ids: List[int] = []
        self._frame_counter = 0

        self.coverage_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self.calibration_result: Optional[Dict] = None

    def add_correspondences(
        self,
        keypoints_cam0: np.ndarray,
        scores_cam0: np.ndarray,
        keypoints_cam1: np.ndarray,
        scores_cam1: np.ndarray,
        min_score: float = 0.5,
    ) -> int:
        """한 프레임의 매칭된 사람 관절 대응점 추가.

        Args:
            keypoints_cam0: (17, 2) 픽셀 좌표 (RTMPose COCO format)
            scores_cam0: (17,) 신뢰도
            keypoints_cam1: (17, 2) 픽셀 좌표
            scores_cam1: (17,) 신뢰도
            min_score: 최소 신뢰도 임계값

        Returns:
            이번 호출에서 추가된 포인트 수
        """
        self._frame_counter += 1
        w, h = self.image_size
        added = 0

        for joint_idx in CALIBRATION_JOINT_INDICES:
            if scores_cam0[joint_idx] < min_score or scores_cam1[joint_idx] < min_score:
                continue

            pt0 = keypoints_cam0[joint_idx]
            pt1 = keypoints_cam1[joint_idx]

            # 이미지 범위 확인
            if not (0 <= pt0[0] < w and 0 <= pt0[1] < h
                    and 0 <= pt1[0] < w and 0 <= pt1[1] < h):
                continue

            self.points_cam0.append(pt0.astype(np.float64).copy())
            self.points_cam1.append(pt1.astype(np.float64).copy())
            self.point_joint_ids.append(joint_idx)
            self.point_frame_ids.append(self._frame_counter)
            added += 1

            # 공간 분포 업데이트 (cam0 기준)
            gx = min(int(pt0[0] / w * GRID_SIZE), GRID_SIZE - 1)
            gy = min(int(pt0[1] / h * GRID_SIZE), GRID_SIZE - 1)
            self.coverage_grid[gy, gx] = True

        return added

    # ------------------------------------------------------------------
    # 상태 조회
    # ------------------------------------------------------------------

    @property
    def num_points(self) -> int:
        return len(self.points_cam0)

    @property
    def coverage_count(self) -> int:
        return int(self.coverage_grid.sum())

    @property
    def progress(self) -> float:
        """캘리브레이션 진행률 0.0 ~ 1.0"""
        point_progress = min(self.num_points / MIN_POINTS, 1.0)
        coverage_progress = min(self.coverage_count / MIN_COVERAGE, 1.0)
        return min(point_progress, coverage_progress)

    def is_ready(self) -> bool:
        """캘리브레이션 가능 여부"""
        return self.num_points >= MIN_POINTS and self.coverage_count >= MIN_COVERAGE

    # ------------------------------------------------------------------
    # 캘리브레이션 실행
    # ------------------------------------------------------------------

    def calibrate(self) -> Optional[Dict]:
        """F → E → R, T 복원 + 인체 비율 스케일 보정

        Returns:
            MotionCapture3D 호환 dict, 또는 실패 시 None
        """
        if not self.is_ready():
            return None

        pts0 = np.array(self.points_cam0, dtype=np.float64)
        pts1 = np.array(self.points_cam1, dtype=np.float64)
        joint_ids = np.array(self.point_joint_ids)
        frame_ids = np.array(self.point_frame_ids)

        # ── Step 1: Fundamental Matrix (RANSAC) ──
        F, mask_f = cv2.findFundamentalMat(
            pts0, pts1, cv2.FM_RANSAC,
            ransacReprojThreshold=3.0, confidence=0.99,
        )
        if F is None:
            return None

        inlier_mask = mask_f.ravel().astype(bool)
        pts0_in = pts0[inlier_mask]
        pts1_in = pts1[inlier_mask]
        joint_ids_in = joint_ids[inlier_mask]
        frame_ids_in = frame_ids[inlier_mask]

        if len(pts0_in) < 8:
            return None

        # ── Step 2: 정규화 → Essential Matrix (RANSAC) ──
        pts0_norm = cv2.undistortPoints(
            pts0_in.reshape(-1, 1, 2), self.K1, None,
        )
        pts1_norm = cv2.undistortPoints(
            pts1_in.reshape(-1, 1, 2), self.K2, None,
        )

        E, mask_e = cv2.findEssentialMat(
            pts0_norm, pts1_norm,
            focal=1.0, pp=(0.0, 0.0),
            method=cv2.RANSAC, prob=0.999, threshold=0.005,
        )
        if E is None:
            return None

        # E 인라이어로 한번 더 필터링
        mask_e_bool = mask_e.ravel().astype(bool)
        pts0_norm_in = pts0_norm[mask_e_bool]
        pts1_norm_in = pts1_norm[mask_e_bool]
        pts0_px_in = pts0_in[mask_e_bool]
        pts1_px_in = pts1_in[mask_e_bool]
        joint_ids_final = joint_ids_in[mask_e_bool]
        frame_ids_final = frame_ids_in[mask_e_bool]

        if len(pts0_norm_in) < 5:
            return None

        # ── Step 3: Recover Pose (R, T) ──
        _, R, T, _ = cv2.recoverPose(E, pts0_norm_in, pts1_norm_in)

        # ── Step 4: 삼각측량 + 인체 스케일 복구 ──
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([R, T])

        points_4d = cv2.triangulatePoints(P1, P2, pts0_px_in.T, pts1_px_in.T)
        points_3d = (points_4d[:3] / points_4d[3:]).T  # (N, 3)

        scale = self._recover_scale(points_3d, joint_ids_final, frame_ids_final)
        T_scaled = T * scale

        # ── 재투영 오차 ──
        reproj_error = self._compute_reprojection_error(
            pts0_px_in, pts1_px_in, R, T,
        )

        self.calibration_result = {
            'camera_matrix_left': self.K1,
            'camera_matrix_right': self.K2,
            'dist_coeffs_left': np.zeros(5),
            'dist_coeffs_right': np.zeros(5),
            'R': R,
            'T': T_scaled,
            'E': E,
            'F': F,
            'image_size': list(self.image_size),
            'reprojection_error': float(reproj_error),
            'num_inliers': int(mask_e_bool.sum()),
            'num_total_points': len(pts0),
            'scale_factor': float(scale),
        }
        return self.calibration_result

    # ------------------------------------------------------------------
    # 스케일 복구
    # ------------------------------------------------------------------

    def _recover_scale(
        self,
        points_3d: np.ndarray,
        joint_ids: np.ndarray,
        frame_ids: np.ndarray,
    ) -> float:
        """인체 세그먼트 길이 중앙값으로 스케일 복구"""
        scales: List[float] = []

        for fid in np.unique(frame_ids):
            fmask = frame_ids == fid
            f_joints = joint_ids[fmask]
            f_pts3d = points_3d[fmask]

            for idx_a, idx_b, true_len in BODY_SEGMENTS:
                mask_a = f_joints == idx_a
                mask_b = f_joints == idx_b
                if mask_a.sum() == 1 and mask_b.sum() == 1:
                    dist = float(np.linalg.norm(f_pts3d[mask_a][0] - f_pts3d[mask_b][0]))
                    if dist > 1e-6:
                        scales.append(true_len / dist)

        if len(scales) >= 3:
            return float(np.median(scales))
        elif scales:
            return float(np.median(scales))
        else:
            # fallback: 어깨 너비 400mm 가정
            return 400.0

    # ------------------------------------------------------------------
    # 재투영 오차
    # ------------------------------------------------------------------

    def _compute_reprojection_error(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
    ) -> float:
        """평균 재투영 오차 (px)"""
        P1 = self.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K2 @ np.hstack([R, T])

        points_4d = cv2.triangulatePoints(P1, P2, pts0.T, pts1.T)
        points_3d = (points_4d[:3] / points_4d[3:]).T

        errors = []
        for i, pt3d in enumerate(points_3d):
            # cam0 투영
            proj0 = self.K1 @ pt3d
            if abs(proj0[2]) < 1e-10:
                continue
            proj0 = proj0[:2] / proj0[2]
            errors.append(float(np.linalg.norm(proj0 - pts0[i])))

            # cam1 투영
            pt_cam1 = R @ pt3d + T.ravel()
            proj1 = self.K2 @ pt_cam1
            if abs(proj1[2]) < 1e-10:
                continue
            proj1 = proj1[:2] / proj1[2]
            errors.append(float(np.linalg.norm(proj1 - pts1[i])))

        return float(np.mean(errors)) if errors else float('inf')

    # ------------------------------------------------------------------
    # 입출력
    # ------------------------------------------------------------------

    def get_calibration_data(self) -> Optional[Dict]:
        """MotionCapture3D 호환 dict 반환"""
        return self.calibration_result

    def save(self, filepath: str):
        """캘리브레이션 결과를 YAML로 저장"""
        if self.calibration_result is None:
            raise ValueError("캘리브레이션 결과가 없습니다")

        data = {}
        for key, val in self.calibration_result.items():
            if isinstance(val, np.ndarray):
                data[key] = val.tolist()
            else:
                data[key] = val

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"[INFO] 캘리브레이션 결과 저장: {filepath}")

    @staticmethod
    def load(filepath: str) -> Dict:
        """YAML에서 캘리브레이션 데이터 로드"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        for key in ['camera_matrix_left', 'camera_matrix_right',
                     'dist_coeffs_left', 'dist_coeffs_right',
                     'R', 'T', 'E', 'F']:
            if key in data and isinstance(data[key], list):
                data[key] = np.array(data[key])

        return data
