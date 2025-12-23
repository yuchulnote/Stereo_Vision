# -*- coding: utf-8 -*-
"""
Queue 성능 테스트 (Step 13)
multiprocessing.Queue의 성능을 측정하여 shared_memory와 비교
"""

import pytest
import multiprocessing as mp
import time
import numpy as np
from queue import Empty


def producer_process(queue: mp.Queue, num_frames: int, width: int, height: int):
    """Producer 프로세스: Queue에 프레임 전송"""
    for i in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        timestamp = time.time_ns()
        queue.put((frame, timestamp, i))
    
    queue.put(None)  # 종료 신호


def consumer_process(queue: mp.Queue):
    """Consumer 프로세스: Queue에서 프레임 수신"""
    count = 0
    while True:
        try:
            item = queue.get(timeout=1.0)
            if item is None:
                break
            frame, timestamp, frame_num = item
            count += 1
        except Empty:
            break
    return count


@pytest.mark.slow
def test_queue_performance():
    """Queue 성능 테스트"""
    width, height = 640, 480
    num_frames = 1000
    
    # Queue 생성
    queue = mp.Queue(maxsize=10)
    
    # 프로세스 시작
    start_time = time.time()
    
    producer = mp.Process(
        target=producer_process,
        args=(queue, num_frames, width, height)
    )
    consumer = mp.Process(target=consumer_process, args=(queue,))
    
    producer.start()
    consumer.start()
    
    producer.join()
    consumer.join()
    
    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    
    print(f"\nQueue 성능 테스트 결과:")
    print(f"  프레임 수: {num_frames}")
    print(f"  소요 시간: {elapsed:.2f}초")
    print(f"  FPS: {fps:.2f}")
    print(f"  프레임당 시간: {elapsed/num_frames*1000:.2f}ms")
    
    # 성능 기준: 최소 30 FPS
    assert fps >= 30, f"Queue 성능이 너무 낮습니다: {fps:.2f} FPS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

