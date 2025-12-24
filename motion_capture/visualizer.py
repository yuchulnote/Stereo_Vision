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


def visualization_process(data_queue: mp.Queue, stop_event: mp.Event):
    """
    Matplotlib 시각화를 담당하는 별도 프로세스 함수
    """
    try:
        # 백엔드 명시적 설정 (Windows 호환성 향상)
        import matplotlib
        # TkAgg나 Qt5Agg가 일반적인 대화형 백엔드
        try:
            matplotlib.use('TkAgg')
        except:
            pass # 기본값 사용
            
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        print("Matplotlib 3D Viewer Process Initializing...")

        # Matplotlib 초기화
        fig = plt.figure(figsize=(10, 8), dpi=100)
        fig.canvas.manager.set_window_title('Stereo Vision 3D Mocap')
        ax = fig.add_subplot(111, projection='3d')
        
        # 깊이(Z) 차이가 잘 보이도록 시점 초기화 (약간 측면에서 바라봄)
        # elev: 위에서 내려다보는 각도, azim: 좌우 회전 각도
        # azim=20 정도로 하면 Depth축이 대각선/가로로 펼쳐져 깊이감이 잘 보임
        ax.view_init(elev=20, azim=20)
        
        plt.ion()
        plt.show() 
        
        # 강제로 이벤트를 처리하여 창을 띄움
        plt.pause(0.1)
        fig.canvas.flush_events()

        print("Matplotlib 3D Viewer Window Opened")

        while not stop_event.is_set():
            try:
                # 큐에서 데이터 가져오기 (타임아웃 0.1초)
                # 큐에 쌓인 최신 데이터만 가져오기 위해 비우기
                data = None
                while True:
                    try:
                        data = data_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if data is None:
                    # 데이터가 없으면 잠시 대기 후 루프 반복
                    plt.pause(0.01)
                    continue
                    
                landmarks_3d, valid_mask = data
                
                # 그리기
                ax.clear()
                ax.set_xlabel('X (Side)')
                ax.set_ylabel('Depth (Z)')
                ax.set_zlabel('Height (-Y)')
                
                # 축 범위 고정 (mm 단위, 자동 스케일링 제거)
                # X: 좌우 3미터 (-1.5m ~ 1.5m)
                ax.set_xlim(-1500, 1500)
                # Y(Depth): 카메라 앞 0 ~ 4미터
                ax.set_ylim(0, 4000)
                # Z(Height): 바닥(-2m) ~ 머리 위(1m)
                ax.set_zlim(-2000, 1000)

                if np.any(valid_mask):
                    # 3D 좌표 변환
                    # OpenCV: X right, Y down, Z forward
                    # Plot: X right, Y depth(forward), Z up(inverted Y)
                    valid_pts = landmarks_3d[valid_mask]
                    xs = valid_pts[:, 0]
                    ys = valid_pts[:, 2]  # Z -> Y (Depth)
                    zs = -valid_pts[:, 1] # Y -> -Z (Height)
                    
                    ax.scatter(xs, ys, zs, c='r', marker='o', s=20)
                    
                    for start_idx, end_idx in MEDIAPIPE_POSE_CONNECTIONS:
                        if (start_idx < len(valid_mask) and end_idx < len(valid_mask) and
                            valid_mask[start_idx] and valid_mask[end_idx]):
                            
                            p1 = landmarks_3d[start_idx]
                            p2 = landmarks_3d[end_idx]
                            
                            lx = [p1[0], p2[0]]
                            ly = [p1[2], p2[2]]
                            lz = [-p1[1], -p2[1]]
                            
                            ax.plot(lx, ly, lz, c='g')
                
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
                
            except Exception as e:
                # 루프 내부 에러는 무시하고 계속 진행
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
        """
        프레임 시각화 및 로깅
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Matplotlib 프로세스로 데이터 전달
        if self.vis_queue is not None:
            try:
                # 큐에 데이터 넣기 (너무 많이 쌓이지 않도록 처리)
                # 시각화 프로세스에서 최신 데이터만 가져가도록 함
                self.vis_queue.put((landmarks_3d, valid_mask))
            except Exception:
                pass
        
        # rerun 시각화
        if self.rerun_initialized and RERUN_AVAILABLE:
            try:
                rr.set_time_seconds("timestamp", timestamp)
                rr.set_time_sequence("frame", self.frame_count)
                
                if np.any(valid_mask):
                    valid_landmarks = landmarks_3d[valid_mask]
                    # ... rerun logging logic (simplified) ...
                    if len(valid_landmarks) > 0:
                        rr.log("world/landmarks", rr.Points3D(positions=valid_landmarks, colors=[[255, 0, 0]] * len(valid_landmarks), radii=[0.02] * len(valid_landmarks)))
                        
                        lines = []
                        for start_idx, end_idx in MEDIAPIPE_POSE_CONNECTIONS:
                            if start_idx < len(valid_mask) and end_idx < len(valid_mask):
                                if valid_mask[start_idx] and valid_mask[end_idx]:
                                    lines.append([landmarks_3d[start_idx], landmarks_3d[end_idx]])
                        
                        if lines:
                            rr.log("world/skeleton", rr.LineStrips3D(strips=lines, colors=[[0, 255, 0]] * len(lines)))
            except Exception as e:
                pass 
        
        # CSV 저장
        if self.export_csv:
            self._save_to_csv(landmarks_3d, valid_mask, timestamp)
        
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
