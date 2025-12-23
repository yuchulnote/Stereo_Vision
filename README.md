# Stereo_Vision

스테레오 비전 프로젝트 - 두 대의 USB 카메라를 사용한 3D 포즈 추정 시스템

## 프로젝트 개요

이 프로젝트는 두 대의 USB 카메라를 사용하여 스테레오 비전 기반 3D 포즈 추정을 수행합니다.

## 프로젝트 구조

```
Stereo_Vision/
├── core/              # 핵심 기능 모듈
├── models/            # 포즈 추정 모델
├── utils/             # 유틸리티 함수
├── calibration/       # 스테레오 카메라 캘리브레이션
├── tests/             # 단위 테스트
├── logs/              # 로그 파일
├── output/            # 출력 파일 (비디오, 이미지)
├── camera_test.py     # 카메라 테스트 스크립트 (Step 1)
├── check_gpu.py       # GPU 환경 확인 스크립트 (Step 4)
├── config.yaml        # 프로젝트 설정 파일 (Step 7)
├── requirements.txt   # Python 패키지 의존성 (Step 3)
├── setup.py            # Cython 컴파일 설정 (Step 10)
├── pytest.ini         # pytest 설정 (Step 9)
└── README.md          # 이 파일
```

## 설치 방법

### 1. Python 가상 환경 생성

```bash
# venv 사용
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 또는 conda 사용
conda create -n stereo_vision python=3.10
conda activate stereo_vision
```

### 2. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 3. GPU 환경 확인

```bash
python check_gpu.py
```

## 사용 방법

### 카메라 테스트

```bash
python camera_test.py
```

### 설정 파일 수정

`config.yaml` 파일을 열어 카메라 설정값을 조정합니다:
- 해상도
- FPS
- 노출, 화이트 밸런스, 포커스 등

## 개발 단계

- ✅ Step 1: OpenCV로 연결된 USB 카메라 2대의 개별 접근 가능 여부 및 최대 해상도/FPS 확인
- ✅ Step 2: 객체 인식/포즈 추정 모델 라이선스 검토 (MediaPipe/RTMPose 선정)
- ✅ Step 3: Python 가상 환경 및 필수 라이브러리 설치
- ✅ Step 4: GPU 가속(CUDA) 환경 구축
- ✅ Step 5: Git 저장소 초기화 및 .gitignore 설정
- ✅ Step 6: 프로젝트 폴더 구조 설계
- ✅ Step 7: 카메라 설정값 고정 (config.yaml)
- ✅ Step 8: 로깅 시스템 구축
- ✅ Step 9: 단위 테스트 프레임워크(pytest) 설정
- ✅ Step 10: 코드 난독화/Cython 컴파일 가능성 검토
- ✅ Step 11-12: 카메라 Producer 프로세스 생성 (multiprocessing)
- ✅ Step 13: Queue 성능 테스트
- ✅ Step 14-15: shared_memory 원형 버퍼 및 Zero-copy 구현
- ✅ Step 16: 세마포어/락 충돌 방지 로직
- ✅ Step 17: 타임스탬프 메타데이터 저장
- ✅ Step 18: CAP_PROP_BUFFERSIZE=1 설정 (셔터 랙 최소화)
- ✅ Step 19-20: 메인 프로세스 뷰어 및 FPS 측정 (목표 60fps)
- ✅ Step 21: 종료 로직 및 atexit 구현 (좀비 프로세스 방지)
- ✅ Step 22: 비디오 녹화 기능
- ✅ Step 23: 자동 재연결 로직
- ✅ Step 24: CPU 사용량 모니터링
- ✅ Step 25: 저조도 환경 테스트

## 스테레오 비전 시스템 실행

### 고성능 멀티프로세싱 시스템

```bash
# 스테레오 비전 시스템 실행
python run_stereo.py
```

자세한 사용법은 `README_STEREO.md`를 참조하세요.

## 테스트 실행

```bash
# 모든 테스트 실행
pytest

# 특정 테스트만 실행
pytest tests/test_camera.py

# 커버리지 포함
pytest --cov
```

## 라이선스

프로젝트 라이선스 정보는 각 모듈의 라이선스를 확인하세요.
- MediaPipe: Apache 2.0
- RTMPose: Apache 2.0

## 주요 모듈

### core/
- `shared_buffer.py`: 공유 메모리 원형 버퍼
- `camera_producer.py`: 카메라 Producer 프로세스
- `stereo_viewer.py`: 스테레오 뷰어 (Consumer)
- `stereo_main.py`: 메인 시스템 관리

### utils/
- `camera_config.py`: 카메라 설정 적용
- `logger.py`: 로깅 시스템

## 참고 문서

- `MODEL_LICENSE_REVIEW.md`: 모델 라이선스 검토 문서
- `OBFUSCATION_REVIEW.md`: 코드 보호 방법 검토 문서
- `README_STEREO.md`: 스테레오 비전 시스템 상세 가이드