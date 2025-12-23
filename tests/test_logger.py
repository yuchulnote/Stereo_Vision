# -*- coding: utf-8 -*-
"""
로거 단위 테스트
"""

import pytest
import sys
from pathlib import Path
import time

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import StereoVisionLogger, PerformanceLogger, get_logger


class TestPerformanceLogger:
    """성능 로거 테스트"""
    
    def test_performance_logger_init(self):
        """성능 로거 초기화 테스트"""
        logger = PerformanceLogger(log_interval=0.1)
        assert logger.log_interval == 0.1
        assert logger.current_fps == 0.0
    
    def test_update_fps(self):
        """FPS 업데이트 테스트"""
        logger = PerformanceLogger()
        logger.update_fps(30.0)
        assert logger.current_fps == 30.0
    
    def test_update_latency(self):
        """지연 시간 업데이트 테스트"""
        logger = PerformanceLogger()
        logger.update_latency(16.5)
        assert logger.current_latency == 16.5
    
    def test_update_sync_error(self):
        """동기화 오차 업데이트 테스트"""
        logger = PerformanceLogger()
        logger.update_sync_error(2.3)
        assert logger.current_sync_error == 2.3
    
    def test_get_stats(self):
        """통계 가져오기 테스트"""
        logger = PerformanceLogger()
        logger.update_fps(30.0)
        logger.update_latency(16.5)
        logger.update_sync_error(2.3)
        
        stats = logger.get_stats()
        assert 'fps' in stats
        assert 'latency_ms' in stats
        assert 'sync_error_ms' in stats
        assert stats['fps'] == 30.0


class TestStereoVisionLogger:
    """스테레오 비전 로거 테스트"""
    
    def test_logger_init(self):
        """로거 초기화 테스트"""
        config = {
            'level': 'INFO',
            'file_path': 'logs/test.log',
            'performance': {
                'log_fps': True,
                'log_latency': True,
                'log_sync_error': True,
                'log_interval_seconds': 0.1
            }
        }
        logger = StereoVisionLogger(config)
        assert logger.logger is not None
    
    def test_logger_singleton(self):
        """로거 싱글톤 테스트"""
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2
    
    def test_log_levels(self):
        """로그 레벨 테스트"""
        logger = get_logger()
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        # 예외가 발생하지 않으면 성공
    
    def test_performance_logging(self):
        """성능 로깅 테스트"""
        logger = get_logger()
        logger.update_fps(30.0)
        logger.update_latency(16.5)
        logger.update_sync_error(2.3)
        logger.log_performance()
        # 예외가 발생하지 않으면 성공


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

