# -*- coding: utf-8 -*-
"""
로깅 시스템
FPS, 처리 지연 시간, 동기화 오차 등을 기록합니다.
"""

import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Optional, Dict
from collections import deque
import threading


class PerformanceLogger:
    """성능 메트릭을 기록하는 클래스"""
    
    def __init__(self, log_interval: float = 1.0):
        """
        Args:
            log_interval: 로그 기록 간격 (초)
        """
        self.log_interval = log_interval
        self.last_log_time = time.time()
        
        # 성능 메트릭 저장
        self.fps_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.sync_error_history = deque(maxlen=100)
        
        # 현재 값
        self.current_fps = 0.0
        self.current_latency = 0.0
        self.current_sync_error = 0.0
        
        # 프레임 카운터
        self.frame_count = 0
        self.start_time = time.time()
        
        # 스레드 안전성
        self.lock = threading.Lock()
    
    def update_fps(self, fps: float):
        """FPS 업데이트"""
        with self.lock:
            self.current_fps = fps
            self.fps_history.append(fps)
            self.frame_count += 1
    
    def update_latency(self, latency_ms: float):
        """처리 지연 시간 업데이트 (밀리초)"""
        with self.lock:
            self.current_latency = latency_ms
            self.latency_history.append(latency_ms)
    
    def update_sync_error(self, error_ms: float):
        """동기화 오차 업데이트 (밀리초)"""
        with self.lock:
            self.current_sync_error = error_ms
            self.sync_error_history.append(error_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """현재 통계 반환"""
        with self.lock:
            return {
                'fps': self.current_fps,
                'latency_ms': self.current_latency,
                'sync_error_ms': self.current_sync_error,
                'frame_count': self.frame_count,
                'avg_fps': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0.0,
                'avg_latency_ms': sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0.0,
                'avg_sync_error_ms': sum(self.sync_error_history) / len(self.sync_error_history) if self.sync_error_history else 0.0,
            }
    
    def should_log(self) -> bool:
        """로그를 기록할 시간인지 확인"""
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.last_log_time = current_time
            return True
        return False


class StereoVisionLogger:
    """스테레오 비전 프로젝트 전용 로거"""
    
    _instance: Optional['StereoVisionLogger'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 로깅 설정 딕셔너리
        """
        if config is None:
            config = {
                'level': 'INFO',
                'file_path': 'logs/stereo_vision.log',
                'max_file_size_mb': 100,
                'backup_count': 5,
                'performance': {
                    'log_fps': True,
                    'log_latency': True,
                    'log_sync_error': True,
                    'log_interval_seconds': 1.0
                }
            }
        
        self.config = config
        self.logger = logging.getLogger('StereoVision')
        self.logger.setLevel(getattr(logging, config.get('level', 'INFO')))
        
        # 기존 핸들러 제거
        self.logger.handlers.clear()
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 파일 핸들러
        file_path = config.get('file_path', 'logs/stereo_vision.log')
        log_dir = Path(file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        max_bytes = config.get('max_file_size_mb', 100) * 1024 * 1024
        backup_count = config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 성능 로거
        perf_config = config.get('performance', {})
        self.performance_logger = PerformanceLogger(
            log_interval=perf_config.get('log_interval_seconds', 1.0)
        )
        
        self.log_fps = perf_config.get('log_fps', True)
        self.log_latency = perf_config.get('log_latency', True)
        self.log_sync_error = perf_config.get('log_sync_error', True)
    
    @classmethod
    def get_instance(cls, config: Optional[Dict] = None) -> 'StereoVisionLogger':
        """싱글톤 인스턴스 반환"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance
    
    def debug(self, message: str):
        """DEBUG 레벨 로그"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """INFO 레벨 로그"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """WARNING 레벨 로그"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERROR 레벨 로그"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """CRITICAL 레벨 로그"""
        self.logger.critical(message)
    
    def log_performance(self):
        """성능 메트릭 로깅"""
        if not self.performance_logger.should_log():
            return
        
        stats = self.performance_logger.get_stats()
        
        if self.log_fps:
            self.info(f"FPS: {stats['fps']:.2f} (평균: {stats['avg_fps']:.2f})")
        
        if self.log_latency:
            self.info(f"처리 지연 시간: {stats['latency_ms']:.2f}ms (평균: {stats['avg_latency_ms']:.2f}ms)")
        
        if self.log_sync_error:
            self.info(f"동기화 오차: {stats['sync_error_ms']:.2f}ms (평균: {stats['avg_sync_error_ms']:.2f}ms)")
        
        self.info(f"총 프레임 수: {stats['frame_count']}")
    
    def update_fps(self, fps: float):
        """FPS 업데이트"""
        self.performance_logger.update_fps(fps)
    
    def update_latency(self, latency_ms: float):
        """처리 지연 시간 업데이트"""
        self.performance_logger.update_latency(latency_ms)
    
    def update_sync_error(self, error_ms: float):
        """동기화 오차 업데이트"""
        self.performance_logger.update_sync_error(error_ms)


def get_logger(config: Optional[Dict] = None) -> StereoVisionLogger:
    """로거 인스턴스 반환"""
    return StereoVisionLogger.get_instance(config)

