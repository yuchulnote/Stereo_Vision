# -*- coding: utf-8 -*-
"""
체스보드 없는 자동 스테레오 캘리브레이션 + 3D 재구성

사람이 걸어다니는 것만으로 캘리브레이션 완료 후 실시간 3D 재구성.
RTMPose (GPU) + OSNet-AIN Re-ID로 여러 사람 환경 지원.

Usage:
    python run_stereo_without_calibration.py
    python run_stereo_without_calibration.py --camera0 0 --camera1 2
    python run_stereo_without_calibration.py --load-calibration calibration_auto_result.yaml
"""

import sys
import os
import argparse
import time
import threading
import queue
from pathlib import Path
from enum import Enum

import cv2
import numpy as np

# PyTorch를 onnxruntime보다 먼저 import (Windows cuDNN DLL 충돌 방지)
import torch  # noqa: F401 — must be before rtmlib/onnxruntime

# ==================== 경로 설정 ====================
STEREO_ROOT = Path(__file__).parent
sys.path.insert(0, str(STEREO_ROOT))

# Overlord 경로 (Re-ID + 트래킹 모듈)
OVERLORD_DIR = os.environ.get(
    'OVERLORD_DIR',
    str(Path(__file__).parent.parent / 'HOSPI_MultiCam' / 'Overlord'),
)
if OVERLORD_DIR not in sys.path:
    sys.path.insert(0, OVERLORD_DIR)

# ==================== 내부 모듈 ====================
from calibration.auto_calibrator import AutoCalibrator
from motion_capture.mocap_3d import MotionCapture3D
from motion_capture.visualizer import MotionCaptureVisualizer

# ==================== Overlord 트래킹 모듈 ====================
from person_tracker import TrackedPerson
from single_camera_tracker import SingleCameraTracker, LocalTrack
from global_identity_manager import GlobalIdentityManager


class StereoGlobalIdentityManager(GlobalIdentityManager):
    """Stereo 전용 GlobalIdentityManager.

    원본은 비겹침 카메라(CCTV) 설계로, "동시 존재 배제" 로직이 있음:
    같은 사람이 두 카메라에 동시에 보이면 → 다른 사람으로 판단 → 매칭 차단.

    Stereo 환경에서는 같은 사람이 두 카메라에 **반드시** 동시에 보이므로
    이 로직을 비활성화한다. _frame_active_gids를 항상 빈 dict로 유지.
    """

    @property
    def _frame_active_gids(self):
        return {}

    @_frame_active_gids.setter
    def _frame_active_gids(self, value):
        pass  # 무시 — 동시 존재 배제 비활성화


# ==================== COCO 17 → MediaPipe 33 변환 ====================
# MotionCapture3D/triangulation.py는 MediaPipe 33-landmark 포맷을 기대
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose
    1: 2,    # L_eye
    2: 5,    # R_eye
    3: 7,    # L_ear
    4: 8,    # R_ear
    5: 11,   # L_shoulder
    6: 12,   # R_shoulder
    7: 13,   # L_elbow
    8: 14,   # R_elbow
    9: 15,   # L_wrist
    10: 16,  # R_wrist
    11: 23,  # L_hip
    12: 24,  # R_hip
    13: 25,  # L_knee
    14: 26,  # R_knee
    15: 27,  # L_ankle
    16: 28,  # R_ankle
}

SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class State(Enum):
    CALIBRATING = "calibrating"
    CALIBRATION_DONE = "calibration_done"
    RECONSTRUCTION_3D = "3d_reconstruction"
    ERROR = "error"


# ==================== Helper Functions ====================

def coco_to_mediapipe_landmarks(
    keypoints: np.ndarray,
    scores: np.ndarray,
    image_size: tuple,
) -> list:
    """COCO 17 keypoints → MediaPipe 33 landmark dicts (정규화 좌표 0-1).

    MotionCapture3D.process()가 기대하는 포맷:
        [{'x': float, 'y': float, 'visibility': float}, ...] × 33
    """
    w, h = image_size
    landmarks = [{'x': 0.0, 'y': 0.0, 'visibility': 0.0} for _ in range(33)]

    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        if coco_idx < len(keypoints):
            landmarks[mp_idx] = {
                'x': float(keypoints[coco_idx][0] / w),
                'y': float(keypoints[coco_idx][1] / h),
                'visibility': float(scores[coco_idx]),
            }

    return landmarks


def compute_bbox(keypoints, scores, frame_shape, threshold=0.5, padding=20):
    """키포인트에서 bbox (x, y, w, h) 계산"""
    fh, fw = frame_shape[:2]
    valid = scores > threshold
    if valid.sum() < 3:
        return None
    valid_pts = keypoints[valid]
    x_min = max(0, int(valid_pts[:, 0].min()) - padding)
    y_min = max(0, int(valid_pts[:, 1].min()) - padding)
    x_max = min(fw, int(valid_pts[:, 0].max()) + padding)
    y_max = min(fh, int(valid_pts[:, 1].max()) + padding)
    if x_max - x_min < 10 or y_max - y_min < 10:
        return None
    return (x_min, y_min, x_max - x_min, y_max - y_min)

# ==================== Drawing ====================

def draw_skeleton(frame, keypoints, scores, color, threshold=0.5):
    """COCO 17 스켈레톤 그리기"""
    for i, (pt, score) in enumerate(zip(keypoints, scores)):
        if score > threshold:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, color, -1)

    for start, end in SKELETON_CONNECTIONS:
        if (start < len(scores) and end < len(scores)
                and scores[start] > threshold and scores[end] > threshold):
            pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
            pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(frame, pt1, pt2, color, 2)


def draw_progress_bar(frame, progress, x, y, width, height):
    """진행률 바 그리기"""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    fill_w = int(width * min(progress, 1.0))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + height), (0, 200, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1)


# ==================== Async Camera ====================

class AsyncCamera:
    """비동기 카메라 캡처 (별도 스레드)"""

    def __init__(self, index, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._running = False
        self._thread = None

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def start(self) -> bool:
        if not self.cap.isOpened():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

    def read(self):
        try:
            return True, self.frame_queue.get(timeout=0.1)
        except queue.Empty:
            return False, None

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()


# ==================== Main Application ====================

class StereoAutoCalibApp:
    """체스보드 없는 자동 캘리브레이션 + 3D 재구성 앱

    상태 머신:
        CALIBRATING → CALIBRATION_DONE → RECONSTRUCTION_3D
                ↑                                       │
                └──── ERROR ← ──────────────────────────┘

    키보드:
        ESC: 종료
        R:   캘리브레이션 리셋/재시도
        C:   수동 캘리브레이션 트리거
    """

    MATCH_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    GRAY = (128, 128, 128)
    AUTO_CALIBRATE_THRESHOLD = 50  # 50포인트 이상이면 자동 캘리브레이션

    # ── 프레임 스킵 설정 (성능 최적화) ──
    # 매 DETECT_INTERVAL 프레임마다 전체 파이프라인 실행,
    # 나머지 프레임은 캐시된 스켈레톤을 현재 카메라 프레임에 오버레이
    DETECT_INTERVAL_3D = 5         # 3D 모드: 5프레임 중 1회 검출
    DETECT_INTERVAL_CALIB = 3      # 캘리브레이션: 3프레임 중 1회 검출
    REID_INTERVAL = 3              # Re-ID는 검출 프레임 중 3회당 1회

    def __init__(
        self,
        camera0_idx: int,
        camera1_idx: int,
        width: int = 640,
        height: int = 480,
        device: str = "cuda",
        rtmpose_mode: str = "balanced",
        reid_model: str = "osnet_ain_x1_0",
        load_calibration: str = None,
    ):
        self.width = width
        self.height = height
        self.image_size = (width, height)
        self.state = State.CALIBRATING
        self.error_msg = ""

        # ── 카메라 ──
        print(f"[INFO] 카메라 초기화: cam0={camera0_idx}, cam1={camera1_idx}")
        self.cam0 = AsyncCamera(camera0_idx, width, height)
        self.cam1 = AsyncCamera(camera1_idx, width, height)

        # ── RTMPose (rtmlib.Body 직접 사용, import chain 회피) ──
        print(f"[INFO] RTMPose 초기화 (mode={rtmpose_mode}, device={device})")
        from rtmlib import Body
        self.body = Body(
            mode=rtmpose_mode,
            backend="onnxruntime",
            device=device,
            det_input_size=(416, 416),
            pose_input_size=(192, 256),
        )
        print("[INFO] RTMPose 초기화 완료")

        # ── Re-ID (optional) ──
        self.reid = None
        try:
            if OVERLORD_DIR not in sys.path:
                sys.path.insert(0, OVERLORD_DIR)
            from reid_extractor import ReIDExtractor
            self.reid = ReIDExtractor(
                model_name=reid_model,
                device=device,
                batch_size=8,
            )
            print(f"[INFO] Re-ID 초기화 완료: {reid_model}")
        except Exception as e:
            print(f"[WARN] Re-ID 초기화 실패 (1명 모드로 동작): {e}")

        # ── Per-camera trackers (Kalman + IoU + Re-ID cascade) ──
        self.tracker0 = SingleCameraTracker(
            camera_id=0, frame_width=width, frame_height=height,
        )
        self.tracker1 = SingleCameraTracker(
            camera_id=1, frame_width=width, frame_height=height,
        )

        # ── Cross-camera global ID manager (stereo 전용: 동시 존재 허용) ──
        self.global_manager = StereoGlobalIdentityManager(
            coord_transformer=None,  # homography 미사용
        )
        self._frame_idx = 0

        # ── Auto Calibrator ──
        self.calibrator = AutoCalibrator(self.image_size)

        # ── 3D 파이프라인 (캘리브레이션 후 초기화) ──
        self._calib_data = None  # 캘리브레이션 결과 (mocap 인스턴스 생성용)
        self.mocap_instances: dict = {}  # {gid: MotionCapture3D}
        self._mocap_last_seen: dict = {}  # {gid: frame_idx} — 정리용
        self.visualizer: MotionCaptureVisualizer = None

        # ── 3D 시점 보정 (카메라 틸트 자동 추정) ──
        self._ground_rotation = np.eye(3)  # 보정 회전행렬
        self._ground_translation = np.zeros(3)  # 바닥 원점 이동
        self._body_up_samples = []  # 인체 수직 벡터 샘플
        self._ground_calibrated = False
        self._GROUND_SAMPLES_NEEDED = 30

        # ── 이전 캘리브레이션 로드 ──
        if load_calibration:
            try:
                calib_data = AutoCalibrator.load(load_calibration)
                self.calibrator.calibration_result = calib_data
                print(f"[INFO] 캘리브레이션 로드 완료: {load_calibration}")
                self.state = State.CALIBRATION_DONE
            except Exception as e:
                print(f"[ERROR] 캘리브레이션 로드 실패: {e}")

        # ── 프레임 스킵 & 캐시 ──
        self._skip_counter = 0
        self._reid_key_counter = 0
        self._cached_draw = None   # 스킵 프레임에서 재사용할 스켈레톤 데이터

        # ── FPS 계산 ──
        self._frame_count = 0
        self._fps_start = time.time()
        self._fps = 0.0

    # ==================================================================
    # 메인 루프
    # ==================================================================

    def run(self):
        if not self.cam0.start():
            print("[ERROR] 카메라 0을 열 수 없습니다")
            return
        if not self.cam1.start():
            print("[ERROR] 카메라 1을 열 수 없습니다")
            self.cam0.stop()
            return

        print("[INFO] 시작! ESC=종료, R=리셋, C=수동 캘리브레이션")

        try:
            while True:
                ret0, frame0 = self.cam0.read()
                ret1, frame1 = self.cam1.read()
                if not ret0 or not ret1:
                    continue

                # 해상도 맞추기
                if frame0.shape[1] != self.width or frame0.shape[0] != self.height:
                    frame0 = cv2.resize(frame0, (self.width, self.height))
                if frame1.shape[1] != self.width or frame1.shape[0] != self.height:
                    frame1 = cv2.resize(frame1, (self.width, self.height))

                # 상태별 처리
                if self.state == State.CALIBRATING:
                    display = self._process_calibrating(frame0, frame1)
                elif self.state == State.CALIBRATION_DONE:
                    self._init_3d_pipeline()
                    display = self._make_side_by_side(
                        frame0, frame1, "3D pipeline initializing...",
                    )
                elif self.state == State.RECONSTRUCTION_3D:
                    display = self._process_3d(frame0, frame1)
                else:  # ERROR
                    display = self._make_side_by_side(
                        frame0, frame1,
                        f"ERROR: {self.error_msg} | R=retry",
                    )

                # FPS
                self._frame_count += 1
                if self._frame_count % 30 == 0:
                    elapsed = time.time() - self._fps_start
                    self._fps = 30 / elapsed if elapsed > 0 else 0
                    self._fps_start = time.time()

                cv2.putText(
                    display, f"FPS: {self._fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )
                cv2.imshow("Stereo Auto-Calibration", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):
                    self._reset()
                elif key == ord('c'):
                    self._manual_calibrate()
        finally:
            self._cleanup()

    # ==================================================================
    # CALIBRATING 상태 처리
    # ==================================================================

    def _process_calibrating(self, frame0, frame1):
        self._skip_counter += 1
        is_key = (self._skip_counter % self.DETECT_INTERVAL_CALIB == 0)

        if is_key:
            # ── KEY FRAME: 전체 파이프라인 ──
            detections0 = self._detect_persons(frame0)
            detections1 = self._detect_persons(frame1)
            matches = self._match_across_cameras(
                frame0, frame1, detections0, detections1,
            )

            # 대응점 수집
            for track0, track1, gid in matches:
                self.calibrator.add_correspondences(
                    track0.keypoints, track0.scores,
                    track1.keypoints, track1.scores,
                )

            # 스켈레톤 데이터 캐시 (스킵 프레임 재사용)
            draw_data = []
            matched_tids_0, matched_tids_1 = set(), set()
            for match_idx, (t0, t1, gid) in enumerate(matches):
                matched_tids_0.add(t0.track_id)
                matched_tids_1.add(t1.track_id)
                draw_data.append((
                    t0.keypoints.copy(), t0.scores.copy(),
                    t1.keypoints.copy(), t1.scores.copy(),
                    t0.bbox, t1.bbox, gid,
                ))

            unmatched_0 = [
                (t.keypoints.copy(), t.scores.copy())
                for t in self.tracker0.get_active_tracks()
                if t.track_id not in matched_tids_0
            ]
            unmatched_1 = [
                (t.keypoints.copy(), t.scores.copy())
                for t in self.tracker1.get_active_tracks()
                if t.track_id not in matched_tids_1
            ]

            self._cached_draw = {
                'matches': draw_data,
                'unmatched_0': unmatched_0,
                'unmatched_1': unmatched_1,
                'num_matches': len(matches),
            }

            # 자동 캘리브레이션 체크
            if self.calibrator.is_ready() and \
               self.calibrator.num_points >= self.AUTO_CALIBRATE_THRESHOLD:
                self._manual_calibrate()

        # ── DRAW (모든 프레임 — 캐시된 스켈레톤 + 현재 카메라 영상) ──
        vis0, vis1 = frame0.copy(), frame1.copy()
        cached = self._cached_draw
        if cached:
            for i, (kp0, sc0, kp1, sc1, bb0, bb1, gid) in enumerate(
                    cached['matches']):
                color = self.MATCH_COLORS[i % len(self.MATCH_COLORS)]
                draw_skeleton(vis0, kp0, sc0, color)
                draw_skeleton(vis1, kp1, sc1, color)
                label = f"GID#{gid}"
                cv2.putText(vis0, label,
                            (int(bb0[0] + bb0[2] / 2) - 20,
                             max(15, bb0[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis1, label,
                            (int(bb1[0] + bb1[2] / 2) - 20,
                             max(15, bb1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for kp, sc in cached['unmatched_0']:
                draw_skeleton(vis0, kp, sc, self.GRAY)
            for kp, sc in cached['unmatched_1']:
                draw_skeleton(vis1, kp, sc, self.GRAY)

        # ── HUD 오버레이 ──
        display = np.hstack([vis0, vis1])
        h_disp, w_disp = display.shape[:2]
        progress = self.calibrator.progress

        bar_y = h_disp - 50
        bar_w = w_disp - 40
        draw_progress_bar(display, progress, 20, bar_y, bar_w, 20)

        n_matches = cached['num_matches'] if cached else 0
        info = (
            f"[CALIBRATING] Points: {self.calibrator.num_points}/30"
            f" | Coverage: {self.calibrator.coverage_count}/6"
            f" | Matches: {n_matches}"
            f" ({'Re-ID' if self.reid else 'Single'})"
        )
        cv2.putText(display, info, (20, bar_y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(display, f"{int(progress * 100)}%",
                     (20 + bar_w // 2 - 15, bar_y + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if self.calibrator.is_ready():
            cv2.putText(
                display, "Ready! Press C or collecting more...",
                (20, bar_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )

        return display

    # ==================================================================
    # RECONSTRUCTION_3D 상태 처리
    # ==================================================================

    def _process_3d(self, frame0, frame1):
        self._skip_counter += 1
        is_key = (self._skip_counter % self.DETECT_INTERVAL_3D == 0)

        if is_key:
            # ── KEY FRAME: 검출 + 매칭 + 3D 재구성 ──
            self._reid_key_counter += 1
            skip_reid = (self._reid_key_counter % self.REID_INTERVAL != 0)

            detections0 = self._detect_persons(frame0)
            detections1 = self._detect_persons(frame1)
            matches = self._match_across_cameras(
                frame0, frame1, detections0, detections1,
                skip_reid=skip_reid,
            )

            all_skeletons = []
            total_joints = 0
            draw_data = []
            matched_tids_0, matched_tids_1 = set(), set()

            for match_idx, (track0, track1, gid) in enumerate(matches):
                matched_tids_0.add(track0.track_id)
                matched_tids_1.add(track1.track_id)

                # 3D 재구성
                lm0 = coco_to_mediapipe_landmarks(
                    track0.keypoints, track0.scores, self.image_size)
                lm1 = coco_to_mediapipe_landmarks(
                    track1.keypoints, track1.scores, self.image_size)

                mocap = self._get_mocap(gid)
                landmarks_3d, valid_mask = mocap.process(lm0, lm1)
                total_joints += int(valid_mask.sum())

                # 시점 보정 수집
                if match_idx == 0 and not self._ground_calibrated:
                    up_vec = self._collect_body_up_vector(
                        landmarks_3d, valid_mask)
                    if up_vec is not None:
                        self._body_up_samples.append(up_vec)
                        if len(self._body_up_samples) >= \
                                self._GROUND_SAMPLES_NEEDED:
                            self._estimate_ground_rotation()

                landmarks_3d = self._apply_ground_correction(
                    landmarks_3d, valid_mask)
                all_skeletons.append((landmarks_3d, valid_mask))

                # 2D 드로우 데이터 캐시
                draw_data.append((
                    track0.keypoints.copy(), track0.scores.copy(),
                    track1.keypoints.copy(), track1.scores.copy(),
                    track0.bbox, track1.bbox, gid,
                ))

            # 3D 시각화 전송
            if self.visualizer and all_skeletons:
                self.visualizer.visualize_frames(all_skeletons)

            self._prune_mocap_instances()

            # 매칭 안 된 트랙
            unmatched_0 = [
                (t.keypoints.copy(), t.scores.copy())
                for t in self.tracker0.get_active_tracks()
                if t.track_id not in matched_tids_0
            ]
            unmatched_1 = [
                (t.keypoints.copy(), t.scores.copy())
                for t in self.tracker1.get_active_tracks()
                if t.track_id not in matched_tids_1
            ]

            self._cached_draw = {
                'matches': draw_data,
                'unmatched_0': unmatched_0,
                'unmatched_1': unmatched_1,
                'total_joints': total_joints,
                'num_matches': len(matches),
            }

        # ── DRAW (모든 프레임 — 캐시된 스켈레톤 + 현재 카메라 프레임) ──
        vis0, vis1 = frame0.copy(), frame1.copy()
        cached = self._cached_draw

        if cached:
            for i, (kp0, sc0, kp1, sc1, bb0, bb1, gid) in enumerate(
                    cached['matches']):
                color = self.MATCH_COLORS[i % len(self.MATCH_COLORS)]
                draw_skeleton(vis0, kp0, sc0, color)
                draw_skeleton(vis1, kp1, sc1, color)
                label = f"GID#{gid}"
                cv2.putText(vis0, label,
                            (int(bb0[0] + bb0[2] / 2) - 20,
                             max(15, bb0[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis1, label,
                            (int(bb1[0] + bb1[2] / 2) - 20,
                             max(15, bb1[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            for kp, sc in cached['unmatched_0']:
                draw_skeleton(vis0, kp, sc, self.GRAY)
            for kp, sc in cached['unmatched_1']:
                draw_skeleton(vis1, kp, sc, self.GRAY)

        display = np.hstack([vis0, vis1])

        reproj = self.calibrator.calibration_result.get(
            'reprojection_error', 0)

        if self._ground_calibrated:
            ground_status = "View OK"
        else:
            ground_status = (
                f"ViewCal {len(self._body_up_samples)}"
                f"/{self._GROUND_SAMPLES_NEEDED}"
            )

        n_matches = cached.get('num_matches', 0) if cached else 0
        total_j = cached.get('total_joints', 0) if cached else 0
        info = (
            f"[3D] Persons: {n_matches}"
            f" | Joints: {total_j}"
            f" | Reproj: {reproj:.2f}px"
            f" | {ground_status}"
        )
        cv2.putText(display, info, (20, display.shape[0] - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display

    # ==================================================================
    # 공통 메서드
    # ==================================================================

    # config와 동일한 검출 임계값 (multi_camera_system과 통일)
    _MIN_VALID_KEYPOINTS = 10
    _KPT_NMS_DIST_THRESH = 15
    _KPT_NMS_MIN_SHARED = 5

    def _detect_persons(self, frame):
        """RTMPose로 프레임에서 사람 검출 + Keypoint NMS.

        multi_camera_system.py의 detect() + _keypoint_nms()와 동일 로직.

        Returns:
            list[TrackedPerson] — Overlord 호환 detection 리스트
        """
        kpts_all, scores_all = self.body(frame)

        detections = []
        for kpts, scrs in zip(kpts_all, scores_all):
            valid = scrs > 0.5
            if valid.sum() < self._MIN_VALID_KEYPOINTS:
                continue
            bbox = compute_bbox(kpts, scrs, frame.shape)
            if bbox is None:
                continue
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            avg_conf = float(scrs[scrs > 0.5].mean()) if (scrs > 0.5).any() else 0.0
            detections.append(TrackedPerson(
                bbox=bbox,
                center=(cx, cy),
                keypoints=kpts,
                scores=scrs,
                confidence=avg_conf,
            ))

        # Keypoint NMS: 동일 인물의 중복 검출 제거
        return self._keypoint_nms(detections)

    @staticmethod
    def _keypoint_nms(detections):
        """같은 index keypoint가 가까운 detection 쌍 → confidence 낮은 쪽 제거."""
        if len(detections) <= 1:
            return detections
        keep = [True] * len(detections)
        dist_thresh = StereoAutoCalibApp._KPT_NMS_DIST_THRESH
        min_shared = StereoAutoCalibApp._KPT_NMS_MIN_SHARED
        for i in range(len(detections)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(detections)):
                if not keep[j]:
                    continue
                kpts_i, kpts_j = detections[i].keypoints, detections[j].keypoints
                scrs_i, scrs_j = detections[i].scores, detections[j].scores
                shared_close = 0
                for k in range(len(kpts_i)):
                    if scrs_i[k] < 0.5 or scrs_j[k] < 0.5:
                        continue
                    if np.linalg.norm(kpts_i[k] - kpts_j[k]) < dist_thresh:
                        shared_close += 1
                if shared_close >= min_shared:
                    if detections[i].confidence < detections[j].confidence:
                        keep[i] = False
                    else:
                        keep[j] = False
        return [d for d, k in zip(detections, keep) if k]

    def _match_across_cameras(self, frame0, frame1,
                               detections0, detections1,
                               skip_reid=False):
        """Overlord 풀 파이프라인으로 크로스-카메라 매칭.

        Args:
            skip_reid: True면 Re-ID 추출을 건너뛰고 zero feature 사용
                       (프레임 스킵 최적화)

        Returns:
            List[(LocalTrack, LocalTrack, int)]
        """
        self._frame_idx += 1
        all_dets = {0: detections0, 1: detections1}
        frames = {0: frame0, 1: frame1}
        trackers = {0: self.tracker0, 1: self.tracker1}

        # ── 색상 히스토그램 계산 (보조 매칭 신호) ──
        for cam_id, dets in all_dets.items():
            f = frames[cam_id]
            fh, fw = f.shape[:2]
            for det in dets:
                x, y, w, h = det.bbox
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x + w), min(fh, y + h)
                if x2 > x1 and y2 > y1:
                    det.color_histogram = self._compute_color_histogram(
                        f[y1:y2, x1:x2])

        # ── 스켈레톤 기반 크롭 + 겹침 마스킹 + Re-ID 추출 ──
        all_features = {0: [None] * len(detections0),
                        1: [None] * len(detections1)}

        if self.reid and not skip_reid:
            all_crops = []
            crop_map = []
            for cam_id, dets in all_dets.items():
                for det_idx, det in enumerate(dets):
                    other_bboxes = [d.bbox for j, d in enumerate(dets)
                                    if j != det_idx]
                    crop = self._get_reid_crop_masked(
                        frames[cam_id], det, other_bboxes)
                    if crop is not None:
                        all_crops.append(crop)
                        crop_map.append((cam_id, det_idx))

            if all_crops:
                extracted = self.reid.extract_from_crops(all_crops)
                for i, (cam_id, det_idx) in enumerate(crop_map):
                    all_features[cam_id][det_idx] = extracted[i]

        # None → zero vector fallback
        for cam_id in (0, 1):
            for i in range(len(all_features[cam_id])):
                if all_features[cam_id][i] is None:
                    all_features[cam_id][i] = np.zeros(512, dtype=np.float32)

        # ── Per-camera tracking (Kalman + IoU + Re-ID cascade) ──
        for cam_id in (0, 1):
            trackers[cam_id].prune_lost_tracks(30.0)
        active0 = self.tracker0.update(detections0, all_features[0])
        active1 = self.tracker1.update(detections1, all_features[1])

        # ── Cross-camera global ID matching ──
        global_mapping = self.global_manager.update(
            camera_tracks={0: active0, 1: active1},
            camera_lost_tracks={
                0: self.tracker0.get_lost_tracks(),
                1: self.tracker1.get_lost_tracks(),
            },
            camera_all_tracks={
                0: self.tracker0.get_all_tracks(),
                1: self.tracker1.get_all_tracks(),
            },
        )

        # ── 빈 장면 감지 → 전체 상태 초기화 ──
        all_empty = (len(self.tracker0.tracks) == 0
                     and len(self.tracker1.tracks) == 0)
        has_state = bool(
            self.global_manager.identities
            or self.global_manager.lost_identities
            or self.global_manager._track_to_global)
        if all_empty and has_state:
            self.tracker0.reset()
            self.tracker1.reset()
            self.global_manager.reset()

        # ── 같은 global_id를 가진 (cam0, cam1) 쌍 찾기 ──
        gid_to_tracks = {}
        track_map0 = {t.track_id: t for t in active0}
        track_map1 = {t.track_id: t for t in active1}

        for (cam_id, tid), gid in global_mapping.items():
            if gid not in gid_to_tracks:
                gid_to_tracks[gid] = {}
            if cam_id == 0 and tid in track_map0:
                gid_to_tracks[gid][0] = track_map0[tid]
            elif cam_id == 1 and tid in track_map1:
                gid_to_tracks[gid][1] = track_map1[tid]

        matches = []
        for gid, cam_tracks in gid_to_tracks.items():
            if 0 in cam_tracks and 1 in cam_tracks:
                matches.append((cam_tracks[0], cam_tracks[1], gid))

        return matches

    # ==================================================================
    # Re-ID 크롭 헬퍼 (multi_camera_system.py에서 이식)
    # ==================================================================

    @staticmethod
    def _compute_color_histogram(crop):
        """HSV 색상 히스토그램 (딥러닝 독립 보조 식별 특징)"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    @staticmethod
    def _crop_body_from_keypoints(frame, keypoints, scores, padding_ratio=0.1):
        """스켈레톤 기반 정밀 크롭 (배경 노이즈 최소화)"""
        h, w = frame.shape[:2]
        valid = scores > 0.3
        if valid.sum() < 4:
            return None
        valid_pts = keypoints[valid]
        min_x, max_x = valid_pts[:, 0].min(), valid_pts[:, 0].max()
        min_y, max_y = valid_pts[:, 1].min(), valid_pts[:, 1].max()
        pw = (max_x - min_x) * padding_ratio
        ph = (max_y - min_y) * padding_ratio
        x1 = int(max(0, min_x - pw))
        y1 = int(max(0, min_y - ph))
        x2 = int(min(w, max_x + pw))
        y2 = int(min(h, max_y + ph))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _get_reid_crop(self, frame, detection):
        """스켈레톤 기반 크롭. Fallback: bbox 크롭."""
        crop = None
        if detection.keypoints is not None and detection.scores is not None:
            crop = self._crop_body_from_keypoints(
                frame, detection.keypoints, detection.scores)
        if crop is None:
            x, y, bw, bh = detection.bbox
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + bw), min(fh, y + bh)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
        return crop

    def _get_reid_crop_masked(self, frame, detection, other_bboxes):
        """다른 사람의 bbox 영역을 마스킹한 후 Re-ID 크롭 추출 (크롭 오염 방지)"""
        if not other_bboxes:
            return self._get_reid_crop(frame, detection)

        dx, dy, dw, dh = detection.bbox
        overlapping = []
        for bbox in other_bboxes:
            ox, oy, ow, oh = bbox
            if (dx < ox + ow and dx + dw > ox and dy < oy + oh and dy + dh > oy):
                overlapping.append(bbox)

        if not overlapping:
            return self._get_reid_crop(frame, detection)

        # 겹치는 영역을 ImageNet 평균색으로 마스킹 (모델 중립 활성화)
        masked_frame = frame.copy()
        fh, fw = frame.shape[:2]
        imagenet_mean_bgr = (104, 116, 124)
        for ox, oy, ow, oh in overlapping:
            mx1, my1 = max(0, ox), max(0, oy)
            mx2, my2 = min(fw, ox + ow), min(fh, oy + oh)
            if mx2 > mx1 and my2 > my1:
                masked_frame[my1:my2, mx1:mx2] = imagenet_mean_bgr

        return self._get_reid_crop(masked_frame, detection)

    # ==================================================================
    # 3D 시점 보정 (카메라 틸트 자동 추정)
    # ==================================================================

    def _collect_body_up_vector(self, landmarks_3d, valid_mask):
        """3D 랜드마크에서 인체 수직 벡터(발목→어깨) 추출.

        MediaPipe 인덱스: L_shoulder=11, R_shoulder=12,
                        L_ankle=27, R_ankle=28
        """
        # 양쪽 어깨와 양쪽 발목이 모두 유효해야 함
        needed = [11, 12, 27, 28]
        if not all(valid_mask[i] for i in needed):
            return None

        shoulder_mid = (landmarks_3d[11] + landmarks_3d[12]) / 2.0
        ankle_mid = (landmarks_3d[27] + landmarks_3d[28]) / 2.0

        up_vec = shoulder_mid - ankle_mid  # 발목 → 어깨 방향
        norm = np.linalg.norm(up_vec)
        if norm < 50.0:  # 50mm 미만이면 무시 (노이즈)
            return None

        return up_vec / norm  # 단위 벡터

    def _estimate_ground_rotation(self):
        """수집된 인체 수직 벡터 샘플로 보정 회전행렬 계산.

        카메라 좌표계에서 인체 수직 방향을 [0, -1, 0] (Y-up)으로 정렬하는
        회전행렬을 Rodrigues 공식으로 계산한다.
        """
        if len(self._body_up_samples) < self._GROUND_SAMPLES_NEEDED:
            return

        # 중앙값 기반 평균 (이상치에 강건)
        samples = np.array(self._body_up_samples)
        avg_up = np.median(samples, axis=0)
        avg_up = avg_up / np.linalg.norm(avg_up)

        # 목표: 인체 수직 벡터 → [0, -1, 0] (카메라 좌표계에서 위쪽)
        target_up = np.array([0.0, -1.0, 0.0])

        # 이미 거의 정렬되어 있으면 보정 불필요
        dot = np.dot(avg_up, target_up)
        if dot > 0.999:
            self._ground_rotation = np.eye(3)
            self._ground_calibrated = True
            print("[INFO] 3D 시점 보정: 카메라가 이미 수평 (보정 불필요)")
            return

        # Rodrigues 회전: avg_up → target_up
        cross = np.cross(avg_up, target_up)
        cross_norm = np.linalg.norm(cross)

        if cross_norm < 1e-6:
            # 반대 방향 (180도 회전) — X축 기준 180도
            self._ground_rotation = np.diag([1.0, -1.0, -1.0])
        else:
            axis = cross / cross_norm
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            # Rodrigues 공식
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            self._ground_rotation = (
                np.eye(3)
                + np.sin(angle) * K
                + (1 - np.cos(angle)) * (K @ K)
            )

        self._ground_calibrated = True

        # 틸트 각도 계산 (사용자 피드백용)
        tilt_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        print(f"[INFO] 3D 시점 보정 완료: 카메라 틸트 ≈ {tilt_deg:.1f}°")

    def _apply_ground_correction(self, landmarks_3d, valid_mask):
        """보정 회전행렬을 3D 랜드마크에 적용.

        Returns:
            보정된 landmarks_3d (원본은 수정하지 않음)
        """
        if not self._ground_calibrated:
            return landmarks_3d

        corrected = landmarks_3d.copy()
        for i in range(len(corrected)):
            if valid_mask[i]:
                corrected[i] = self._ground_rotation @ corrected[i]

        return corrected

    # ==================================================================
    # 3D 파이프라인 초기화
    # ==================================================================

    def _init_3d_pipeline(self):
        """캘리브레이션 결과로 3D 파이프라인 초기화"""
        try:
            calib_data = self.calibrator.get_calibration_data()
            if calib_data is None:
                self.state = State.ERROR
                self.error_msg = "calibration data is None"
                return

            self._calib_data = calib_data
            self.mocap_instances = {}
            self._mocap_last_seen = {}
            self._cached_draw = None  # 캘리브레이션 캐시 초기화
            self._skip_counter = 0

            self.visualizer = MotionCaptureVisualizer(use_matplotlib=True)

            self.state = State.RECONSTRUCTION_3D
            print("[INFO] 3D 파이프라인 초기화 완료")

        except Exception as e:
            self.state = State.ERROR
            self.error_msg = str(e)
            import traceback
            traceback.print_exc()

    def _get_mocap(self, gid: int) -> MotionCapture3D:
        """GID별 MotionCapture3D 인스턴스 (on-demand 생성)"""
        if gid not in self.mocap_instances:
            self.mocap_instances[gid] = MotionCapture3D(
                calibration_data=self._calib_data,
                confidence_threshold=0.5,
                use_midpoint=True,
                filter_enabled=True,
            )
        self._mocap_last_seen[gid] = self._frame_idx
        return self.mocap_instances[gid]

    def _prune_mocap_instances(self, max_age: int = 300):
        """오래 안 보인 GID의 mocap 인스턴스 정리 (메모리)"""
        stale = [
            gid for gid, last in self._mocap_last_seen.items()
            if self._frame_idx - last > max_age
        ]
        for gid in stale:
            del self.mocap_instances[gid]
            del self._mocap_last_seen[gid]

    def _manual_calibrate(self):
        """캘리브레이션 실행"""
        if not self.calibrator.is_ready():
            print(
                f"[WARN] 포인트 부족: {self.calibrator.num_points}/30, "
                f"커버리지: {self.calibrator.coverage_count}/6"
            )
            return

        print("[INFO] 캘리브레이션 시작...")
        result = self.calibrator.calibrate()

        if result is None:
            print("[ERROR] 캘리브레이션 실패 (RANSAC)")
            self.state = State.ERROR
            self.error_msg = "Calibration failed (RANSAC)"
            return

        reproj = result['reprojection_error']
        inliers = result['num_inliers']
        scale = result['scale_factor']
        print(f"[INFO] 캘리브레이션 완료!")
        print(f"  Reprojection error: {reproj:.2f}px")
        print(f"  Inliers: {inliers}/{result['num_total_points']}")
        print(f"  Scale factor: {scale:.2f}")

        save_path = str(STEREO_ROOT / "calibration_auto_result.yaml")
        self.calibrator.save(save_path)

        self.state = State.CALIBRATION_DONE

    def _reset(self):
        """캘리브레이션 리셋"""
        print("[INFO] 캘리브레이션 리셋")
        self.calibrator.reset()
        self.mocap_instances = {}
        self._mocap_last_seen = {}
        self._calib_data = None
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
        # 트래커/매니저 초기화
        self.tracker0 = SingleCameraTracker(
            camera_id=0,
            frame_width=self.width,
            frame_height=self.height,
        )
        self.tracker1 = SingleCameraTracker(
            camera_id=1,
            frame_width=self.width,
            frame_height=self.height,
        )
        self.global_manager = StereoGlobalIdentityManager(coord_transformer=None)
        self._frame_idx = 0
        # 3D 시점 보정 초기화
        self._ground_rotation = np.eye(3)
        self._ground_translation = np.zeros(3)
        self._body_up_samples = []
        self._ground_calibrated = False
        # 프레임 스킵 초기화
        self._skip_counter = 0
        self._reid_key_counter = 0
        self._cached_draw = None
        self.state = State.CALIBRATING
        self.error_msg = ""

    def _make_side_by_side(self, frame0, frame1, text=""):
        display = np.hstack([frame0, frame1])
        if text:
            cv2.putText(display, text, (20, display.shape[0] - 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return display

    def _cleanup(self):
        self.cam0.stop()
        self.cam1.stop()
        if self.visualizer:
            self.visualizer.close()
        cv2.destroyAllWindows()
        print("[INFO] 종료")


# ==================== 카메라 선택 ====================

def _get_camera_names() -> dict:
    """카메라 디바이스 이름 가져오기 (인덱스 → 이름)"""
    names = {}
    # pygrabber (DirectShow 열거 — 가장 정확한 인덱스 매핑)
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        for i, name in enumerate(devices):
            names[i] = name
        return names
    except ImportError:
        pass
    # PowerShell fallback
    try:
        import subprocess
        result = subprocess.run(
            ['powershell', '-Command',
             'Get-PnpDevice -Class Camera -Status OK '
             '| Select-Object -ExpandProperty FriendlyName; '
             'Get-PnpDevice -Class Image -Status OK '
             '| Select-Object -ExpandProperty FriendlyName'],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        if result.returncode == 0 and result.stdout.strip():
            for i, name in enumerate(result.stdout.strip().split('\n')):
                if name.strip():
                    names[i] = name.strip()
    except Exception:
        pass
    return names


def select_cameras():
    """Brio 500 카메라 2대 자동 선택. 없으면 대화형 선택."""
    print("\n사용 가능한 카메라 검색 중...")
    camera_names = _get_camera_names()

    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = camera_names.get(i, f"Camera {i}")
            available.append((i, name, f"{w}x{h}"))
            cap.release()

    if len(available) < 2:
        print(f"[ERROR] 카메라 2대 이상 필요 (현재: {len(available)}대)")
        sys.exit(1)

    print(f"\n사용 가능한 카메라: {len(available)}대")
    for idx, (cam_id, name, res) in enumerate(available):
        print(f"  [{idx}] Camera {cam_id}: {name} ({res})")

    # Brio 500 자동 선택
    brio_indices = [
        cam_id for cam_id, name, _ in available
        if 'brio' in name.lower() and '500' in name
    ]
    if len(brio_indices) >= 2:
        cam0, cam1 = brio_indices[0], brio_indices[1]
        print(f"\n[INFO] Brio 500 2대 자동 선택: cam0={cam0}, cam1={cam1}")
        return cam0, cam1

    print("[WARN] Brio 500 2대를 찾지 못했습니다. 수동 선택해주세요.")

    while True:
        try:
            c0 = int(input("\n카메라 0 선택 (번호): "))
            if 0 <= c0 < len(available):
                break
        except ValueError:
            pass
        print("올바른 번호를 입력하세요")

    while True:
        try:
            c1 = int(input("카메라 1 선택 (번호): "))
            if 0 <= c1 < len(available) and c1 != c0:
                break
        except ValueError:
            pass
        print("다른 번호를 입력하세요")

    return available[c0][0], available[c1][0]


# ==================== Entry Point ====================

def main():
    parser = argparse.ArgumentParser(
        description="체스보드 없는 자동 스테레오 캘리브레이션 + 3D 재구성",
    )
    parser.add_argument("--camera0", type=int, default=None,
                        help="카메라 0 인덱스")
    parser.add_argument("--camera1", type=int, default=None,
                        help="카메라 1 인덱스")
    parser.add_argument("--width", type=int, default=640,
                        help="프레임 너비")
    parser.add_argument("--height", type=int, default=480,
                        help="프레임 높이")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--rtmpose-mode", type=str, default="balanced",
                        choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--reid-model", type=str, default="osnet_ain_x1_0",
                        help="Re-ID 모델 이름")
    parser.add_argument("--load-calibration", type=str, default=None,
                        help="이전 캘리브레이션 결과 YAML 경로")

    args = parser.parse_args()

    print("=" * 60)
    print("  Stereo Auto-Calibration (RTMPose + Re-ID)")
    print("  체스보드 없이 사람 관절로 자동 캘리브레이션")
    print("=" * 60)

    # 카메라 선택
    if args.camera0 is None or args.camera1 is None:
        cam0, cam1 = select_cameras()
    else:
        cam0, cam1 = args.camera0, args.camera1

    app = StereoAutoCalibApp(
        camera0_idx=cam0,
        camera1_idx=cam1,
        width=args.width,
        height=args.height,
        device=args.device,
        rtmpose_mode=args.rtmpose_mode,
        reid_model=args.reid_model,
        load_calibration=args.load_calibration,
    )
    app.run()


if __name__ == "__main__":
    main()
