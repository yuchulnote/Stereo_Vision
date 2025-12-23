
import numpy as np
from scipy import signal
from typing import List, Tuple, Optional
import logging

class TemporalSynchronizer:
    """
    두 카메라의 시간 동기화를 위한 클래스.
    MediaPipe Pose 랜드마크(손목 등)의 수직 움직임(속도)을 상호 상관(Cross-Correlation)하여 
    시간 오차(Lag)를 계산합니다.
    """

    def __init__(self, fps: float = 30.0, smoothing_sigma: float = 2.0):
        """
        Args:
            fps: 카메라 FPS (기본값: 30.0)
            smoothing_sigma: Gaussian Smoothing 시그마 값 (노이즈 제거용)
        """
        self.fps = fps
        self.smoothing_sigma = smoothing_sigma
        self.logger = logging.getLogger("TemporalSync")

    def _gaussian_smooth(self, data: np.ndarray) -> np.ndarray:
        """Gaussian Smoothing 적용 (Step 15)"""
        window_size = int(6 * self.smoothing_sigma + 1)
        if window_size % 2 == 0:
            window_size += 1
        
        # scipy.ndimage.gaussian_filter1d 대신 간단한 convolution 사용 가능
        # 여기서는 scipy.signal.windows.gaussian 사용
        window = signal.windows.gaussian(window_size, std=self.smoothing_sigma)
        window /= window.sum()
        
        return np.convolve(data, window, mode='same')

    def _calculate_velocity(self, data: np.ndarray) -> np.ndarray:
        """위치를 속도로 변환 (Step 16)"""
        # v_t = y_t - y_{t-1}
        # np.diff는 길이가 1 줄어드므로 앞에 0 추가하여 길이 유지
        velocity = np.diff(data, prepend=data[0])
        return velocity

    def _normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """Z-score 정규화 (Step 17)"""
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - np.mean(data)) / std

    def calculate_time_offset(self, 
                            signal1: List[float], 
                            signal2: List[float]) -> Tuple[float, float, Tuple[np.ndarray, np.ndarray]]:
        """
        두 신호 간의 시간 오차 계산
        
        Args:
            signal1: 카메라 1의 신호 (예: Y좌표 시계열)
            signal2: 카메라 2의 신호
            
        Returns:
            offset_ms: 시간 오차 (ms), 양수면 signal2가 signal1보다 늦음 (Lag)
            max_corr: 최대 상관계수 (신뢰도)
            (proc_sig1, proc_sig2): 처리된 신호 (시각화용)
        """
        # numpy 배열 변환
        s1 = np.array(signal1)
        s2 = np.array(signal2)
        
        # 길이가 다르면 짧은 쪽에 맞춤
        min_len = min(len(s1), len(s2))
        s1 = s1[:min_len]
        s2 = s2[:min_len]
        
        # 움직임 강도 체크 (위치 데이터의 표준편차)
        std1 = np.std(s1)
        std2 = np.std(s2)
        if std1 < 0.02 or std2 < 0.02: # 0~1 사이의 Y좌표. 0.02 미만이면 거의 가만히 있는 것
            self.logger.warning(f"Low movement detected in position data! std1={std1:.4f}, std2={std2:.4f}")
            # 움직임이 너무 적으면 실패 처리 (낮은 상관계수 반환)
            return 0.0, -1.0, (np.array([]), np.array([]))
        
        # 1. 노이즈 제거 (Gaussian Smoothing)
        s1_smooth = self._gaussian_smooth(s1)
        s2_smooth = self._gaussian_smooth(s2)
        
        # 2. 속도 변환 (Velocity)
        v1 = self._calculate_velocity(s1_smooth)
        v2 = self._calculate_velocity(s2_smooth)
        
        # 3. 속도 신호의 변동성 체크 (이것이 더 중요함!)
        # 움직임이 없으면 속도가 거의 0이 되어 정규화 후에도 신뢰할 수 없는 결과가 나옴
        v1_std = np.std(v1)
        v2_std = np.std(v2)
        
        # 속도 표준편차가 매우 작으면 (거의 움직임이 없으면) 실패 처리
        # 위치가 0~1 범위이고 프레임 간 변화가 0.001 미만이면 거의 정지 상태
        if v1_std < 0.001 or v2_std < 0.001:
            self.logger.warning(f"Low movement detected in velocity data! v1_std={v1_std:.6f}, v2_std={v2_std:.6f}")
            return 0.0, -1.0, (v1, v2)
        
        # 4. 정규화 (Normalization)
        v1_norm = self._normalize_signal(v1)
        v2_norm = self._normalize_signal(v2)
        
        # 정규화 후에도 표준편차 체크 (0이면 모두 같은 값 = 움직임 없음)
        if np.std(v1_norm) < 0.1 or np.std(v2_norm) < 0.1:
            self.logger.warning(f"Normalized signal has low variance! v1_std={np.std(v1_norm):.4f}, v2_std={np.std(v2_norm):.4f}")
            return 0.0, -1.0, (v1_norm, v2_norm)
        
        # 5. 상호 상관 (Cross-Correlation) - Step 18
        # mode='full'을 사용하여 모든 가능한 lag 계산
        correlation = signal.correlate(v1_norm, v2_norm, mode='full')
        lags = signal.correlation_lags(len(v1_norm), len(v2_norm), mode='full')
        
        # 6. Peak 검출 (Step 19)
        peak_idx = np.argmax(correlation)
        lag_frames = lags[peak_idx]
        max_corr = correlation[peak_idx] / min_len  # 정규화된 상관계수 (대략 -1 ~ 1)
        
        # 7. 시간 변환 (Step 20)
        # lag_frames > 0 이면 s1이 s2보다 앞서 있음 (s2가 지연됨) -> s2 timestamp에서 빼야 함
        # 예: lag=5 (s1이 5프레임 오른쪽으로 이동해야 겹침 -> s1이 5프레임 먼저 나옴)
        # 수식: Time_Offset = Lag * (1000 / FPS)
        offset_ms = lag_frames * (1000.0 / self.fps)
        
        self.logger.info(f"Sync Result: Lag={lag_frames} frames, Offset={offset_ms:.2f}ms, MaxCorr={max_corr:.3f}, v1_std={v1_std:.6f}, v2_std={v2_std:.6f}")
        
        return offset_ms, max_corr, (v1_norm, v2_norm)

