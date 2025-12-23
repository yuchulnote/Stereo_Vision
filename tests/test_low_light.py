# -*- coding: utf-8 -*-
"""
저조도 환경 테스트 (Step 25)
노이즈 레벨 확인
"""

import pytest
import cv2
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.camera_config import open_camera_with_config


@pytest.mark.camera
def test_low_light_noise():
    """저조도 환경에서 노이즈 레벨 테스트"""
    cap = open_camera_with_config(0)
    
    if cap is None:
        pytest.skip("카메라를 사용할 수 없습니다")
    
    try:
        # 여러 프레임 캡처
        frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        if len(frames) < 5:
            pytest.skip("충분한 프레임을 캡처할 수 없습니다")
        
        # 평균 프레임 계산
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        
        # 노이즈 추정: 각 프레임과 평균 프레임의 차이
        noise_levels = []
        for frame in frames:
            diff = np.abs(frame.astype(np.float32) - avg_frame.astype(np.float32))
            noise_level = np.mean(diff)
            noise_levels.append(noise_level)
        
        avg_noise = np.mean(noise_levels)
        max_noise = np.max(noise_levels)
        
        print(f"\n저조도 노이즈 테스트 결과:")
        print(f"  평균 노이즈 레벨: {avg_noise:.2f}")
        print(f"  최대 노이즈 레벨: {max_noise:.2f}")
        
        # 노이즈 레벨이 너무 높으면 경고
        if avg_noise > 20:
            print(f"  ⚠️  노이즈 레벨이 높습니다. 노출 설정을 조정하세요.")
        
        # 기본 검증: 노이즈 레벨이 합리적인 범위 내에 있는지
        assert avg_noise < 50, f"노이즈 레벨이 너무 높습니다: {avg_noise:.2f}"
        
    finally:
        cap.release()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "camera"])

