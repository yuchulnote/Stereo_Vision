# -*- coding: utf-8 -*-
"""
공유 메모리 원형 버퍼 (Ring Buffer)
multiprocessing.shared_memory를 사용하여 프로세스 간 이미지 데이터 공유
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Tuple, Optional
import time
import struct


class SharedRingBuffer:
    """공유 메모리 원형 버퍼 클래스"""
    
    def __init__(
        self,
        buffer_size: int,
        width: int,
        height: int,
        channels: int = 3,
        dtype: np.dtype = np.uint8,
        name: Optional[str] = None
    ):
        """
        Args:
            buffer_size: 버퍼 슬롯 개수 (원형 버퍼 크기)
            width: 이미지 너비
            height: 이미지 높이
            channels: 채널 수 (기본값: 3 = RGB)
            dtype: 데이터 타입 (기본값: uint8)
            name: 공유 메모리 이름 (None이면 자동 생성)
        """
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.channels = channels
        self.dtype = dtype
        
        # 이미지 데이터 크기
        self.frame_size = width * height * channels * np.dtype(dtype).itemsize
        
        # 메타데이터 크기 (타임스탬프 + 프레임 번호)
        self.metadata_size = 16  # time.time_ns() (8 bytes) + frame_number (8 bytes)
        
        # 슬롯 크기 (이미지 + 메타데이터)
        self.slot_size = self.frame_size + self.metadata_size
        
        # 전체 공유 메모리 크기
        self.total_size = buffer_size * self.slot_size
        
        # 공유 메모리 생성 또는 연결
        if name is None:
            # 새 공유 메모리 생성
            self.shm = shared_memory.SharedMemory(create=True, size=self.total_size)
            self.is_creator = True
        else:
            # 기존 공유 메모리 연결
            self.shm = shared_memory.SharedMemory(name=name)
            self.is_creator = False
        
        self.name = self.shm.name
        
        # 공유 메모리를 numpy 배열로 뷰 생성
        self.buffer = np.ndarray(
            (buffer_size, self.slot_size),
            dtype=np.uint8,
            buffer=self.shm.buf
        )
        
        # 쓰기 인덱스 (Producer용)
        self.write_index = mp.Value('i', 0)
        
        # 읽기 인덱스 (Consumer용)
        self.read_index = mp.Value('i', 0)
        
        # 세마포어: 쓰기 가능 여부
        self.write_semaphore = mp.Semaphore(buffer_size)
        
        # 세마포어: 읽기 가능 여부
        self.read_semaphore = mp.Semaphore(0)
        
        # 락: 인덱스 업데이트 보호
        self.write_lock = mp.Lock()
        self.read_lock = mp.Lock()
    
    def write_frame(self, frame: np.ndarray, timestamp_ns: int, frame_number: int) -> bool:
        """
        프레임을 버퍼에 쓰기
        
        Args:
            frame: 이미지 프레임 (numpy 배열)
            timestamp_ns: 타임스탬프 (나노초)
            frame_number: 프레임 번호
            
        Returns:
            성공 여부
        """
        # 프레임 크기 확인
        if frame.shape != (self.height, self.width, self.channels):
            # 디버깅 정보
            import sys
            print(
                f"[DEBUG] 프레임 크기 불일치: "
                f"프레임={frame.shape}, 버퍼=({self.height}, {self.width}, {self.channels})",
                file=sys.stderr
            )
            return False
        
        # 쓰기 가능한 슬롯 대기 (Non-blocking, timeout=0)
        # 버퍼가 가득 차면 오래된 프레임을 덮어쓰기 (최신 프레임 우선)
        acquired = self.write_semaphore.acquire(timeout=0.0)
        buffer_full = not acquired
        
        try:
            with self.write_lock:
                # 현재 쓰기 인덱스
                idx = self.write_index.value
                
                # 버퍼가 가득 찬 경우, 오래된 프레임 건너뛰기
                if buffer_full:
                    with self.read_lock:
                        if self.read_index.value == idx:
                            # 읽기 인덱스가 쓰기 인덱스와 같으면 (모든 슬롯 사용 중)
                            # 오래된 프레임 건너뛰기
                            self.read_index.value = (idx + 1) % self.buffer_size
                            # 쓰기 세마포어 해제 (덮어쓰기로 인해 슬롯이 해제됨)
                            try:
                                self.write_semaphore.release()
                            except ValueError:
                                pass
                
                # 이미지 데이터 복사
                # frame.flatten()은 복사본을 생성할 수 있으므로, 버퍼에 직접 할당
                # np.copyto가 더 효율적일 수 있음
                target_buffer = self.buffer[idx, :self.frame_size]
                frame_flat = frame.ravel()  # ravel은 가능한 경우 뷰를 반환 (더 빠름)
                
                # 데이터 크기 재확인
                if frame_flat.nbytes != target_buffer.nbytes:
                    import sys
                    print(f"[ERROR] 데이터 크기 불일치: {frame_flat.nbytes} vs {target_buffer.nbytes}", file=sys.stderr)
                else:
                    # 버퍼에 데이터 복사
                    np.copyto(target_buffer, frame_flat)
                
                # 메타데이터 저장 (타임스탬프 + 프레임 번호)
                metadata = struct.pack('QQ', timestamp_ns, frame_number)
                metadata_array = np.frombuffer(metadata, dtype=np.uint8)
                self.buffer[idx, self.frame_size:self.slot_size] = metadata_array
                
                # 인덱스 업데이트 (원형 버퍼)
                self.write_index.value = (idx + 1) % self.buffer_size
            
            # 읽기 가능 신호
            try:
                self.read_semaphore.release()
            except ValueError:
                # 세마포어가 이미 최대값인 경우 (읽기가 따라가지 못함)
                # 이는 정상적인 상황일 수 있음 (Consumer가 느린 경우)
                pass
            
            return True
            
        except Exception as e:
            # 오류 발생 시 세마포어 해제
            if acquired:
                self.write_semaphore.release()
            return False
    
    def read_frame(self) -> Optional[Tuple[np.ndarray, int, int]]:
        """
        버퍼에서 프레임 읽기
        
        Returns:
            (frame, timestamp_ns, frame_number) 또는 None
        """
        # 읽기 가능한 슬롯 대기 (Non-blocking)
        # timeout을 약간 늘려 읽기 성공률 향상
        acquired = self.read_semaphore.acquire(timeout=0.1)
        if not acquired:
            return None
        
        try:
            with self.read_lock:
                # 현재 읽기 인덱스
                idx = self.read_index.value
                
                # 이미지 데이터 복사
                frame_data = self.buffer[idx, :self.frame_size].copy()
                frame = frame_data.reshape(self.height, self.width, self.channels)
                
                # 메타데이터 읽기
                metadata = bytes(self.buffer[idx, self.frame_size:self.slot_size])
                timestamp_ns, frame_number = struct.unpack('QQ', metadata)
                
                # 인덱스 업데이트 (원형 버퍼)
                self.read_index.value = (idx + 1) % self.buffer_size
            
            # 쓰기 가능 신호
            self.write_semaphore.release()
            
            return frame, timestamp_ns, frame_number
            
        except Exception as e:
            self.read_semaphore.release()
            return None
    
    def ensure_mapping(self):
        """
        프로세스 간 객체 전달 시 numpy 배열의 메모리 매핑이 끊어질 수 있으므로
        수신 측 프로세스에서 이 메서드를 호출하여 매핑을 복구해야 합니다.
        """
        # 기존 buffer가 유효한지 확인하고, 필요시 재생성
        try:
            # shm.buf 접근 시도
            _ = self.shm.buf[0]
        except:
            pass

        # numpy 배열 재생성 (shm.buf에 다시 매핑)
        # 중요: pickle된 객체에서 shm.buf는 유효한 mmap 핸들을 가리키고 있어야 함
        self.buffer = np.ndarray(
            (self.buffer_size, self.slot_size),
            dtype=np.uint8,
            buffer=self.shm.buf
        )

    def close(self):
        """공유 메모리 닫기"""
        if hasattr(self, 'shm'):
            self.shm.close()
            if self.is_creator:
                self.shm.unlink()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

