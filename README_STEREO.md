# 스테레오 비전 시스템 사용 가이드

## 개요

이 시스템은 multiprocessing과 shared_memory를 사용하여 두 대의 USB 카메라로부터 고성능 영상 캡처 및 표시를 수행합니다.

## 주요 기능

### Step 11-12: 카메라 Producer 프로세스
- 각 카메라를 별도 프로세스에서 캡처
- 프로세스 간 독립성 보장

### Step 13: Queue 성능 테스트
- `tests/test_queue_performance.py`에서 Queue 성능 측정
- shared_memory와 성능 비교 가능

### Step 14-15: 공유 메모리 원형 버퍼
- `core/shared_buffer.py`: Zero-copy 지향 이미지 공유
- 원형 버퍼로 메모리 효율성 향상

### Step 16: 세마포어/락 충돌 방지
- 쓰기/읽기 세마포어로 동기화
- 락으로 인덱스 업데이트 보호

### Step 17: 타임스탬프 메타데이터
- 각 프레임에 `time.time_ns()` 타임스탬프 저장
- 동기화 오차 측정 가능

### Step 18: 셔터 랙 최소화
- `CAP_PROP_BUFFERSIZE=1` 설정
- 최신 프레임만 유지

### Step 19-20: 뷰어 및 FPS 측정
- 좌우 결합 영상 표시
- 실시간 FPS 측정 (목표: 60fps)

### Step 21: 종료 로직
- `atexit` 및 시그널 핸들러로 정리
- 좀비 프로세스 방지

### Step 22: 비디오 녹화
- OpenCV VideoWriter 사용
- 'r' 키로 녹화 토글

### Step 23: 자동 재연결
- 카메라 연결 끊김 시 자동 재연결
- 재연결 간격 설정 가능

### Step 24: CPU 모니터링
- psutil을 사용한 CPU 사용률 모니터링
- 별도 스레드에서 실행

### Step 25: 저조도 테스트
- `tests/test_low_light.py`에서 노이즈 레벨 확인

## 사용 방법

### 1. 환경 설정

```bash
# 가상 환경 활성화
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 필수 라이브러리 설치
pip install -r requirements.txt
```

### 2. 설정 파일 수정

`config.yaml`에서 카메라 설정을 확인하세요:
- `camera_0`: 첫 번째 카메라 설정
- `camera_1`: 두 번째 카메라 설정

### 3. 시스템 실행

#### 대화형 카메라 선택 (권장)

```bash
python run_stereo.py
```

실행 시 사용 가능한 카메라 목록이 표시되고, 카메라 0과 카메라 1을 각각 선택할 수 있습니다.

#### 명령줄 인자로 카메라 지정

```bash
# 카메라 인덱스 직접 지정
python run_stereo.py --camera0 0 --camera1 1

# 카메라 0만 지정 (카메라 1은 선택 화면)
python run_stereo.py --camera0 0

# 설정 파일 지정
python run_stereo.py --config my_config.yaml
```

또는

```bash
python -m core.stereo_main
```

### 4. 키보드 단축키

- `q`: 프로그램 종료
- `r`: 비디오 녹화 토글

### 5. 성능 모니터링

화면에 표시되는 정보:
- 각 카메라의 FPS
- CPU 사용률
- 동기화 오차 (밀리초)

## 테스트

### Queue 성능 테스트

```bash
pytest tests/test_queue_performance.py -v -s
```

### 저조도 환경 테스트

```bash
pytest tests/test_low_light.py -v -s -m camera
```

## 성능 최적화 팁

1. **해상도 조정**: `config.yaml`에서 해상도를 낮추면 FPS 향상
2. **버퍼 크기**: `core/stereo_main.py`의 `buffer_size` 조정
3. **프로세스 우선순위**: Windows에서 프로세스 우선순위 설정 가능

## 문제 해결

### 카메라가 열리지 않는 경우
- 카메라 인덱스 확인: `python camera_test.py`
- 다른 프로그램에서 카메라 사용 중인지 확인
- 카메라 선택 화면에서 올바른 인덱스 선택 확인

### 카메라 선택 화면이 나타나지 않는 경우
- 명령줄 인자로 카메라 인덱스를 직접 지정: `--camera0`, `--camera1`
- `config.yaml`에서 카메라 인덱스 확인

### FPS가 낮은 경우
- 해상도 낮추기
- 다른 프로세스의 CPU 사용 확인
- GPU 가속 확인: `python check_gpu.py`

### 메모리 오류
- 버퍼 크기 줄이기
- 해상도 낮추기

## 아키텍처

```
메인 프로세스 (Consumer/Viewer)
    ├── Producer 프로세스 0 (카메라 0)
    │   └── SharedMemory Buffer 0
    ├── Producer 프로세스 1 (카메라 1)
    │   └── SharedMemory Buffer 1
    └── Viewer (화면 표시)
```

## 참고 사항

- Windows에서는 `multiprocessing`의 `spawn` 방법 사용
- 공유 메모리는 프로세스 종료 시 자동 정리
- CPU 모니터링은 `psutil` 라이브러리 필요

