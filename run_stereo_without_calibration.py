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


def match_persons_reid(features_0, features_1, threshold=0.4):
    """Re-ID 피처로 크로스-카메라 매칭 (Hungarian algorithm).

    Returns:
        List[(idx_cam0, idx_cam1)] 매칭된 인덱스 쌍
    """
    if not features_0 or not features_1:
        return []

    feat_0 = np.array(features_0)  # (N, 512)
    feat_1 = np.array(features_1)  # (M, 512)

    # cosine similarity (L2 정규화 → dot product = cosine sim)
    sim_matrix = feat_0 @ feat_1.T
    cost_matrix = 1.0 - sim_matrix

    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost_matrix)

    matches = []
    for r, c in zip(rows, cols):
        if sim_matrix[r, c] >= threshold:
            matches.append((int(r), int(c)))

    return matches


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

    def __init__(
        self,
        camera0_idx: int,
        camera1_idx: int,
        width: int = 640,
        height: int = 480,
        device: str = "cuda",
        rtmpose_mode: str = "balanced",
        reid_model: str = "osnet_ain_x0_25",
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

        # ── Auto Calibrator ──
        self.calibrator = AutoCalibrator(self.image_size)

        # ── 3D 파이프라인 (캘리브레이션 후 초기화) ──
        self.mocap: MotionCapture3D = None
        self.visualizer: MotionCaptureVisualizer = None

        # ── 이전 캘리브레이션 로드 ──
        if load_calibration:
            try:
                calib_data = AutoCalibrator.load(load_calibration)
                self.calibrator.calibration_result = calib_data
                print(f"[INFO] 캘리브레이션 로드 완료: {load_calibration}")
                self.state = State.CALIBRATION_DONE
            except Exception as e:
                print(f"[ERROR] 캘리브레이션 로드 실패: {e}")

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
        # ── 포즈 검출 ──
        persons0, bboxes0 = self._detect_persons(frame0)
        persons1, bboxes1 = self._detect_persons(frame1)

        # ── Re-ID 크로스-카메라 매칭 ──
        matches = self._match_across_cameras(
            frame0, frame1, persons0, bboxes0, persons1, bboxes1,
        )

        # ── 대응점 수집 + 시각화 ──
        vis0, vis1 = frame0.copy(), frame1.copy()

        for match_idx, (i0, i1) in enumerate(matches):
            color = self.MATCH_COLORS[match_idx % len(self.MATCH_COLORS)]
            p0, p1 = persons0[i0], persons1[i1]

            self.calibrator.add_correspondences(
                p0['keypoints'], p0['scores'],
                p1['keypoints'], p1['scores'],
            )

            draw_skeleton(vis0, p0['keypoints'], p0['scores'], color)
            draw_skeleton(vis1, p1['keypoints'], p1['scores'], color)

            # 매칭 번호
            bx0, by0 = int(p0['bbox'][0] + p0['bbox'][2] / 2) - 30, p0['bbox'][1] - 10
            bx1, by1 = int(p1['bbox'][0] + p1['bbox'][2] / 2) - 30, p1['bbox'][1] - 10
            cv2.putText(vis0, f"Match#{match_idx+1}", (bx0, by0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis1, f"Match#{match_idx+1}", (bx1, by1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 매칭 안 된 사람 (회색)
        matched_0 = {m[0] for m in matches}
        matched_1 = {m[1] for m in matches}
        for i, p in enumerate(persons0):
            if i not in matched_0:
                draw_skeleton(vis0, p['keypoints'], p['scores'], self.GRAY)
        for i, p in enumerate(persons1):
            if i not in matched_1:
                draw_skeleton(vis1, p['keypoints'], p['scores'], self.GRAY)

        # ── 합성 + 오버레이 ──
        display = np.hstack([vis0, vis1])
        h_disp = display.shape[0]
        w_disp = display.shape[1]
        progress = self.calibrator.progress

        # 진행률 바
        bar_y = h_disp - 50
        bar_w = w_disp - 40
        draw_progress_bar(display, progress, 20, bar_y, bar_w, 20)

        info = (
            f"[CALIBRATING] Points: {self.calibrator.num_points}/30"
            f" | Coverage: {self.calibrator.coverage_count}/6"
            f" | Matches: {len(matches)}"
            f" ({'Re-ID' if self.reid else 'Single'})"
        )
        cv2.putText(display, info, (20, bar_y - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        pct_text = f"{int(progress * 100)}%"
        cv2.putText(display, pct_text, (20 + bar_w // 2 - 15, bar_y + 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ready 상태 표시
        if self.calibrator.is_ready():
            cv2.putText(
                display, "Ready! Press C to calibrate or collecting more...",
                (20, bar_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            # 충분히 모이면 자동 캘리브레이션
            if self.calibrator.num_points >= self.AUTO_CALIBRATE_THRESHOLD:
                self._manual_calibrate()

        return display

    # ==================================================================
    # RECONSTRUCTION_3D 상태 처리
    # ==================================================================

    def _process_3d(self, frame0, frame1):
        persons0, bboxes0 = self._detect_persons(frame0)
        persons1, bboxes1 = self._detect_persons(frame1)

        matches = self._match_across_cameras(
            frame0, frame1, persons0, bboxes0, persons1, bboxes1,
        )

        vis0, vis1 = frame0.copy(), frame1.copy()
        valid_count = 0

        # 첫 번째 매칭 쌍으로 3D 재구성
        if matches and self.mocap:
            i0, i1 = matches[0]
            p0, p1 = persons0[i0], persons1[i1]

            # COCO 17 → MediaPipe 33 변환 (정규화 좌표)
            lm0 = coco_to_mediapipe_landmarks(
                p0['keypoints'], p0['scores'], self.image_size,
            )
            lm1 = coco_to_mediapipe_landmarks(
                p1['keypoints'], p1['scores'], self.image_size,
            )

            landmarks_3d, valid_mask = self.mocap.process(lm0, lm1)
            valid_count = int(valid_mask.sum())

            if self.visualizer:
                self.visualizer.visualize_frame(landmarks_3d, valid_mask)

            # 메인 매칭 스켈레톤 (녹색)
            draw_skeleton(vis0, p0['keypoints'], p0['scores'], (0, 255, 0))
            draw_skeleton(vis1, p1['keypoints'], p1['scores'], (0, 255, 0))

        # 나머지 사람 (회색)
        matched_0 = {m[0] for m in matches}
        matched_1 = {m[1] for m in matches}
        for i, p in enumerate(persons0):
            if i not in matched_0:
                draw_skeleton(vis0, p['keypoints'], p['scores'], self.GRAY)
        for i, p in enumerate(persons1):
            if i not in matched_1:
                draw_skeleton(vis1, p['keypoints'], p['scores'], self.GRAY)

        display = np.hstack([vis0, vis1])

        reproj = self.calibrator.calibration_result.get('reprojection_error', 0)
        info = (
            f"[3D] Matches: {len(matches)}"
            f" | Joints: {valid_count}"
            f" | Reproj: {reproj:.2f}px"
        )
        cv2.putText(display, info, (20, display.shape[0] - 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return display

    # ==================================================================
    # 공통 메서드
    # ==================================================================

    def _detect_persons(self, frame):
        """RTMPose로 프레임에서 사람 검출.

        Returns:
            (persons, bboxes) — persons: list[dict], bboxes: list[tuple]
        """
        kpts_all, scores_all = self.body(frame)

        persons = []
        bboxes = []
        for kpts, scrs in zip(kpts_all, scores_all):
            valid = scrs > 0.5
            if valid.sum() < 5:
                continue
            bbox = compute_bbox(kpts, scrs, frame.shape)
            if bbox is None:
                continue
            persons.append({'keypoints': kpts, 'scores': scrs, 'bbox': bbox})
            bboxes.append(bbox)

        return persons, bboxes

    def _match_across_cameras(self, frame0, frame1,
                               persons0, bboxes0, persons1, bboxes1):
        """크로스-카메라 매칭. Re-ID 없으면 1명 전용."""
        if not persons0 or not persons1:
            return []

        if self.reid:
            features0 = self.reid.extract(frame0, bboxes0)
            features1 = self.reid.extract(frame1, bboxes1)
            return match_persons_reid(features0, features1)

        # Re-ID 없으면: 양쪽 1명씩일 때만 매칭
        if len(persons0) == 1 and len(persons1) == 1:
            return [(0, 0)]

        return []

    def _init_3d_pipeline(self):
        """캘리브레이션 결과로 3D 파이프라인 초기화"""
        try:
            calib_data = self.calibrator.get_calibration_data()
            if calib_data is None:
                self.state = State.ERROR
                self.error_msg = "calibration data is None"
                return

            self.mocap = MotionCapture3D(
                calibration_data=calib_data,
                confidence_threshold=0.5,
                use_midpoint=True,
                filter_enabled=True,
            )

            self.visualizer = MotionCaptureVisualizer(use_matplotlib=True)

            self.state = State.RECONSTRUCTION_3D
            print("[INFO] 3D 파이프라인 초기화 완료")

        except Exception as e:
            self.state = State.ERROR
            self.error_msg = str(e)
            import traceback
            traceback.print_exc()

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
        self.mocap = None
        if self.visualizer:
            self.visualizer.close()
            self.visualizer = None
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
    parser.add_argument("--reid-model", type=str, default="osnet_ain_x0_25",
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
