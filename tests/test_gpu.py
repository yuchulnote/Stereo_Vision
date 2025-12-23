# -*- coding: utf-8 -*-
"""
GPU 환경 확인 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestGPUEnvironment:
    """GPU 환경 테스트"""
    
    @pytest.mark.gpu
    def test_pytorch_cuda_available(self):
        """PyTorch CUDA 사용 가능 여부 테스트"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            assert isinstance(cuda_available, bool)
            if cuda_available:
                assert torch.cuda.device_count() > 0
        except ImportError:
            pytest.skip("PyTorch가 설치되지 않았습니다")
    
    @pytest.mark.gpu
    def test_onnxruntime_providers(self):
        """ONNX Runtime 실행 공급자 테스트"""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            assert isinstance(providers, list)
            assert len(providers) > 0
        except ImportError:
            pytest.skip("ONNX Runtime이 설치되지 않았습니다")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

