# -*- coding: utf-8 -*-
"""
3D 모션캡쳐 시각화 및 데이터 로깅
rerun-sdk 또는 Matplotlib를 활용한 실시간 3D 스켈레톤 시각화
별도 프로세스(multiprocessing)에서 시각화를 수행하여 GUI 충돌 방지
"""

import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import sys
import csv
from datetime import datetime
import time
import multiprocessing as mp
import queue

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# rerun은 사용자가 삭제했으므로 임포트 시도만 남겨두거나 완전히 제거할 수 있음
# 코드 호환성을 위해 변수는 유지
RERUN_AVAILABLE = False
try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from utils.logger import get_logger

# MediaPipe Pose 연결 정보
MEDIAPIPE_POSE_CONNECTIONS = [
    # 얼굴
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    # 상체
    (9, 10),  # 어깨
    (11, 12),  # 어깨-팔꿈치
    (11, 13), (12, 14),  # 팔꿈치-손목
    (13, 15), (14, 16),  # 손목-손
    (11, 23), (12, 24),  # 어깨-골반
    (23, 24),  # 골반
    # 하체
    (23, 25), (24, 26),  # 골반-무릎
    (25, 27), (26, 28),  # 무릎-발목
    (27, 29), (28, 30),  # 발목-발가락
    (29, 31), (30, 32),  # 발가락
]


# 사람별 색상 (matplotlib named colors)
PERSON_COLORS = [
    ('#00ff00', '#22cc22'),  # 녹색 (관절, 뼈)
    ('#ff4444', '#cc2222'),  # 빨강
    ('#4488ff', '#2266cc'),  # 파랑
    ('#ffaa00', '#cc8800'),  # 주황
    ('#ff44ff', '#cc22cc'),  # 분홍
    ('#00cccc', '#009999'),  # 청록
]


def visualization_process(data_queue: mp.Queue, stop_event: mp.Event):
    """
    Matplotlib 시각화를 담당하는 별도 프로세스 함수.

    큐 데이터 형식:
      - 단일: (landmarks_3d, valid_mask)             ← 하위 호환
      - 다중: [(landmarks_3d, valid_mask), ...]       ← 여러 사람
    """
    try:
        import matplotlib
        try:
            matplotlib.use('TkAgg')
        except Exception:
            pass

        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        print("Matplotlib 3D Viewer Process Initializing...")

        fig = plt.figure(figsize=(10, 8), dpi=100)
        fig.canvas.manager.set_window_title('Stereo Vision 3D Mocap')
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=20)

        plt.ion()
        plt.show()
        plt.pause(0.1)
        fig.canvas.flush_events()

        print("Matplotlib 3D Viewer Window Opened")

        # EMA 상태 (축 범위 스무딩)
        smooth_cx, smooth_cy, smooth_cz = 0.0, 2000.0, 0.0
        smooth_range = 1000.0
        ema_alpha = 0.15  # 낮을수록 부드러움
        initialized = False

        while not stop_event.is_set():
            try:
                # 큐에 쌓인 최신 데이터만 사용
                data = None
                while True:
                    try:
                        data = data_queue.get_nowait()
                    except queue.Empty:
                        break

                if data is None:
                    plt.pause(0.01)
                    continue

                # ── 데이터 정규화: 항상 list of (landmarks_3d, valid_mask) ──
                if isinstance(data, list):
                    skeletons = data
                else:
                    # 하위 호환: 단일 (landmarks_3d, valid_mask) 튜플
                    skeletons = [data]

                # ── 전체 유효 포인트 수집 (auto-center/scale 용) ──
                all_plot_pts = []
                for landmarks_3d, valid_mask in skeletons:
                    if np.any(valid_mask):
                        vp = landmarks_3d[valid_mask]
                        # OpenCV→Plot: X=X, Y=Z(depth), Z=-Y(height)
                        pts = np.column_stack([vp[:, 0], vp[:, 2], -vp[:, 1]])
                        all_plot_pts.append(pts)

                # ── 그리기 ──
                ax.clear()
                ax.set_xlabel('X (Side)')
                ax.set_ylabel('Depth (Z)')
                ax.set_zlabel('Height (-Y)')

                if all_plot_pts:
                    all_pts = np.vstack(all_plot_pts)

                    # 중심 & 범위 계산
                    cx = (all_pts[:, 0].min() + all_pts[:, 0].max()) / 2
                    cy = (all_pts[:, 1].min() + all_pts[:, 1].max()) / 2
                    cz = (all_pts[:, 2].min() + all_pts[:, 2].max()) / 2
                    span = max(
                        all_pts[:, 0].max() - all_pts[:, 0].min(),
                        all_pts[:, 1].max() - all_pts[:, 1].min(),
                        all_pts[:, 2].max() - all_pts[:, 2].min(),
                        400.0,  # 최소 400mm
                    ) / 2 * 1.5  # 50% 패딩

                    # EMA 스무딩 (축이 프레임마다 튀지 않도록)
                    if not initialized:
                        smooth_cx, smooth_cy, smooth_cz = cx, cy, cz
                        smooth_range = span
                        initialized = True
                    else:
                        smooth_cx += ema_alpha * (cx - smooth_cx)
                        smooth_cy += ema_alpha * (cy - smooth_cy)
                        smooth_cz += ema_alpha * (cz - smooth_cz)
                        smooth_range += ema_alpha * (span - smooth_range)

                    ax.set_xlim(smooth_cx - smooth_range, smooth_cx + smooth_range)
                    ax.set_ylim(smooth_cy - smooth_range, smooth_cy + smooth_range)
                    ax.set_zlim(smooth_cz - smooth_range, smooth_cz + smooth_range)

                    # ── 사람별 스켈레톤 그리기 ──
                    for person_idx, (landmarks_3d, valid_mask) in enumerate(skeletons):
                        if not np.any(valid_mask):
                            continue

                        ci = person_idx % len(PERSON_COLORS)
                        joint_color = PERSON_COLORS[ci][0]
                        bone_color = PERSON_COLORS[ci][1]

                        vp = landmarks_3d[valid_mask]
                        xs = vp[:, 0]
                        ys = vp[:, 2]
                        zs = -vp[:, 1]
                        ax.scatter(xs, ys, zs, c=joint_color, marker='o', s=25)

                        for start_idx, end_idx in MEDIAPIPE_POSE_CONNECTIONS:
                            if (start_idx < len(valid_mask) and
                                    end_idx < len(valid_mask) and
                                    valid_mask[start_idx] and valid_mask[end_idx]):
                                p1 = landmarks_3d[start_idx]
                                p2 = landmarks_3d[end_idx]
                                ax.plot(
                                    [p1[0], p2[0]],
                                    [p1[2], p2[2]],
                                    [-p1[1], -p2[1]],
                                    c=bone_color, linewidth=2,
                                )
                else:
                    # 데이터 없으면 기본 범위
                    ax.set_xlim(-1000, 1000)
                    ax.set_ylim(0, 3000)
                    ax.set_zlim(-1500, 1000)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

            except Exception:
                pass

    except Exception as e:
        print(f"Visualization process error: {e}")

    plt.close('all')
    print("Matplotlib 3D Viewer Process Ended")


class MotionCaptureVisualizer:
    """3D 모션캡쳐 시각화 및 로깅 클래스"""
    
    def __init__(
        self,
        recording_path: Optional[str] = None,
        export_csv: bool = True,
        export_bvh: bool = False,
        csv_path: Optional[str] = None,
        use_matplotlib: bool = False
    ):
        """
        Args:
            recording_path: rerun recording 경로 (None이면 실시간 스트리밍)
            export_csv: CSV 내보내기 활성화
            export_bvh: BVH 내보내기 활성화
            csv_path: CSV 저장 경로
            use_matplotlib: Matplotlib 실시간 시각화 사용 여부
        """
        self.export_csv = export_csv
        self.export_bvh = export_bvh
        self.csv_path = csv_path or "output/mocap_data.csv"
        self.use_matplotlib = use_matplotlib
        
        self.logger = get_logger()
        
        # rerun 초기화
        self.rerun_initialized = False
        if RERUN_AVAILABLE and not self.use_matplotlib:
            try:
                if recording_path:
                    rr.init("MotionCapture3D", recording_path=recording_path)
                else:
                    rr.init("MotionCapture3D", spawn=True)
                self.rerun_initialized = True
                self.logger.info("rerun-sdk 초기화 완료 - 3D 시각화 활성화")
                
                # 초기 뷰 설정
                try:
                    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, timeless=True)
                except:
                    pass
            except Exception as e:
                self.logger.warning(f"rerun-sdk 초기화 실패: {e}")
        
        # Matplotlib 프로세스 초기화
        self.vis_process = None
        self.vis_queue = None
        self.vis_stop_event = None
        
        if self.use_matplotlib and MATPLOTLIB_AVAILABLE:
            try:
                self.vis_queue = mp.Queue()
                self.vis_stop_event = mp.Event()
                self.vis_process = mp.Process(
                    target=visualization_process,
                    args=(self.vis_queue, self.vis_stop_event),
                    daemon=True
                )
                self.vis_process.start()
                self.logger.info("Matplotlib 3D 시각화 프로세스 시작")
            except Exception as e:
                self.logger.warning(f"Matplotlib 프로세스 시작 실패: {e}")

        # CSV 파일 초기화
        if self.export_csv:
            self._init_csv()
        
        # 프레임 카운터
        self.frame_count = 0
    
    def _init_csv(self):
        """CSV 파일 초기화"""
        csv_dir = Path(self.csv_path).parent
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['frame', 'timestamp']
            for i in range(33):
                header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z', f'landmark_{i}_valid'])
            writer.writerow(header)
    
    def visualize_frame(
        self,
        landmarks_3d: np.ndarray,
        valid_mask: np.ndarray,
        timestamp: Optional[float] = None
    ):
        """단일 사람 프레임 시각화 (하위 호환)"""
        self.visualize_frames([(landmarks_3d, valid_mask)], timestamp)

    def visualize_frames(
        self,
        skeletons: List[Tuple[np.ndarray, np.ndarray]],
        timestamp: Optional[float] = None
    ):
        """
        여러 사람의 3D 스켈레톤 시각화 및 로깅.

        Args:
            skeletons: [(landmarks_3d, valid_mask), ...] 리스트
            timestamp: 타임스탬프
        """
        if timestamp is None:
            timestamp = time.time()

        # Matplotlib 프로세스로 데이터 전달 (리스트 형식)
        if self.vis_queue is not None:
            try:
                self.vis_queue.put(skeletons)
            except Exception:
                pass

        # rerun 시각화 (첫 번째 사람만 — rerun은 기존 로직 유지)
        if self.rerun_initialized and RERUN_AVAILABLE and skeletons:
            try:
                landmarks_3d, valid_mask = skeletons[0]
                rr.set_time_seconds("timestamp", timestamp)
                rr.set_time_sequence("frame", self.frame_count)

                if np.any(valid_mask):
                    valid_landmarks = landmarks_3d[valid_mask]
                    if len(valid_landmarks) > 0:
                        rr.log("world/landmarks", rr.Points3D(
                            positions=valid_landmarks,
                            colors=[[255, 0, 0]] * len(valid_landmarks),
                            radii=[0.02] * len(valid_landmarks)))

                        lines = []
                        for start_idx, end_idx in MEDIAPIPE_POSE_CONNECTIONS:
                            if (start_idx < len(valid_mask) and
                                    end_idx < len(valid_mask) and
                                    valid_mask[start_idx] and valid_mask[end_idx]):
                                lines.append([landmarks_3d[start_idx],
                                              landmarks_3d[end_idx]])
                        if lines:
                            rr.log("world/skeleton", rr.LineStrips3D(
                                strips=lines,
                                colors=[[0, 255, 0]] * len(lines)))
            except Exception:
                pass

        # CSV 저장 (첫 번째 사람)
        if self.export_csv and skeletons:
            self._save_to_csv(skeletons[0][0], skeletons[0][1], timestamp)

        self.frame_count += 1
    
    def _save_to_csv(self, landmarks_3d: np.ndarray, valid_mask: np.ndarray, timestamp: float):
        """CSV 파일에 저장"""
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [self.frame_count, timestamp]
                for i in range(len(landmarks_3d)):
                    if valid_mask[i]:
                        row.extend([landmarks_3d[i][0], landmarks_3d[i][1], landmarks_3d[i][2], 1])
                    else:
                        row.extend([0.0, 0.0, 0.0, 0])
                writer.writerow(row)
        except Exception:
            pass
    
    def close(self):
        """리소스 정리"""
        if self.rerun_initialized and RERUN_AVAILABLE:
            try:
                rr.disconnect()
            except:
                pass
        
        # Matplotlib 프로세스 종료
        if self.vis_stop_event is not None:
            self.vis_stop_event.set()
        
        if self.vis_process is not None:
            try:
                self.vis_process.join(timeout=1.0)
                if self.vis_process.is_alive():
                    self.vis_process.terminate()
            except:
                pass
