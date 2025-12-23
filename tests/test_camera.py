# -*- coding: utf-8 -*-
"""
카메라 관련 단위 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import camera_test


class TestCameraFunctions:
    """카메라 함수 테스트"""
    
    def test_get_camera_names_windows(self):
        """Windows에서 카메라 이름 가져오기 테스트"""
        import platform
        if platform.system() == 'Windows':
            names = camera_test.get_camera_names_windows()
            assert isinstance(names, list)
    
    def test_get_camera_info(self):
        """카메라 정보 가져오기 테스트"""
        info = camera_test.get_camera_info(0)
        assert isinstance(info, dict)
        assert 'index' in info
        assert 'name' in info
        assert 'backend' in info
        assert 'available' in info
    
    def test_test_camera_index(self):
        """카메라 인덱스 테스트"""
        result = camera_test.test_camera_index(0)
        assert isinstance(result, bool)
    
    @pytest.mark.camera
    def test_get_available_cameras(self):
        """사용 가능한 카메라 찾기 테스트"""
        available, info_dict = camera_test.get_available_cameras(max_index=3)
        assert isinstance(available, list)
        assert isinstance(info_dict, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

