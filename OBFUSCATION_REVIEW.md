# 코드 난독화 및 컴파일 가능성 검토

## 검토 일자
2024년

## 검토 목적
상용 배포를 고려한 코드 보호 방법 검토 (난독화 및 Cython 컴파일)

## 1. 코드 난독화 (Obfuscation)

### 1.1 Python 난독화 도구

#### pyarmor
- **라이선스**: 상용 (무료 버전 제한 있음)
- **특징**:
  - 바이트코드 암호화
  - 실행 시 복호화
  - 상용 라이선스 필요 시 비용 발생
- **장점**: 강력한 보호
- **단점**: 라이선스 비용, 성능 오버헤드

#### pyobfuscate
- **라이선스**: 오픈소스
- **특징**:
  - 변수명/함수명 변경
  - 코드 구조 변경
- **장점**: 무료
- **단점**: 보호 수준 낮음, 유지보수 어려움

#### py-minifier
- **라이선스**: 오픈소스
- **특징**:
  - 코드 압축 및 최소화
  - 변수명 단축
- **장점**: 무료, 간단
- **단점**: 보호 수준 매우 낮음

### 1.2 난독화의 한계

1. **완전한 보호 불가능**: Python은 해석형 언어이므로 완전한 보호는 어렵습니다.
2. **성능 오버헤드**: 암호화/복호화 과정으로 인한 성능 저하
3. **디버깅 어려움**: 오류 발생 시 디버깅이 매우 어려워짐
4. **유지보수 복잡**: 난독화된 코드는 유지보수가 어렵습니다.

## 2. Cython 컴파일

### 2.1 Cython 개요

- **라이선스**: Apache 2.0
- **특징**:
  - Python 코드를 C로 컴파일
  - .pyx 파일을 .so/.pyd로 컴파일
  - 성능 향상 + 어느 정도의 코드 보호

### 2.2 Cython 사용 방법

#### 설치
```bash
pip install cython
```

#### 기본 사용법
1. `.py` 파일을 `.pyx`로 변환
2. `setup.py` 작성
3. 컴파일: `python setup.py build_ext --inplace`

#### 예시: setup.py
```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "core/stereo_vision.py",
        "utils/logger.py",
    ], compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
    })
)
```

### 2.3 Cython의 장단점

#### 장점
- ✅ 성능 향상 (특히 계산 집약적 코드)
- ✅ 어느 정도의 코드 보호 (소스 코드 숨김)
- ✅ C 확장 모듈로 배포 가능
- ✅ 기존 Python 코드와 호환

#### 단점
- ⚠️ 컴파일 시간 소요
- ⚠️ 플랫폼별 빌드 필요 (Windows, Linux, macOS)
- ⚠️ 디버깅이 어려워짐
- ⚠️ 완전한 보호는 아님 (역공학 가능)

## 3. 권장 사항

### 3.1 단계별 접근

#### Phase 1: 개발 단계
- 난독화 사용 안 함
- 코드 가독성과 디버깅 용이성 우선

#### Phase 2: 배포 준비 단계
- **핵심 알고리즘만 Cython으로 컴파일**
  - `core/` 모듈의 계산 집약적 부분
  - `calibration/` 모듈
- **설정 및 유틸리티는 Python 유지**
  - `utils/logger.py` (로깅은 디버깅 필요)
  - `config.yaml` (설정 파일)

#### Phase 3: 상용 배포 단계
- Cython 컴파일 + 최소한의 난독화 (선택사항)
- 라이선스 검증 로직 추가
- 실행 파일 패키징 (PyInstaller, cx_Freeze 등)

### 3.2 구체적 구현 계획

#### Cython 컴파일 대상
```
core/
  ├── stereo_vision.pyx  (컴파일)
  ├── depth_estimation.pyx  (컴파일)
  └── calibration.pyx  (컴파일)

calibration/
  └── stereo_calib.pyx  (컴파일)
```

#### Python 유지 대상
```
utils/
  ├── logger.py  (Python 유지 - 디버깅 필요)
  └── camera_utils.py  (Python 유지)

config.yaml  (설정 파일)
```

### 3.3 빌드 시스템

#### setup.py 예시
```python
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "core.stereo_vision",
        ["core/stereo_vision.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3'],
    ),
    # 추가 모듈...
]

setup(
    name="stereo_vision",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
    zip_safe=False,
)
```

#### 빌드 명령
```bash
# 개발용 (디버그 정보 포함)
python setup.py build_ext --inplace

# 배포용 (최적화)
python setup.py build_ext --inplace --optimize
```

## 4. 대안: 실행 파일 패키징

### 4.1 PyInstaller
- **특징**: Python 애플리케이션을 단일 실행 파일로 패키징
- **장점**: 사용자에게 Python 설치 불필요
- **단점**: 파일 크기 증가, 바이러스 백신 오탐 가능

### 4.2 cx_Freeze
- **특징**: 크로스 플랫폼 실행 파일 생성
- **장점**: 무료, 오픈소스
- **단점**: 설정 복잡

## 5. 결론 및 권장사항

### 최종 권장 사항

1. **즉시 적용 가능**: 
   - Cython으로 핵심 알고리즘 컴파일
   - 성능 향상 + 어느 정도의 코드 보호

2. **선택적 적용**:
   - 상용 배포 시 PyInstaller로 실행 파일 패키징
   - 라이선스 검증 로직 추가

3. **비권장**:
   - 과도한 난독화 (유지보수 어려움)
   - 모든 코드 난독화 (디버깅 불가능)

### 구현 우선순위

1. ✅ **Cython 컴파일 시스템 구축** (높은 우선순위)
2. ⚠️ **PyInstaller 패키징** (중간 우선순위)
3. ❌ **난독화** (낮은 우선순위, 필요 시에만)

## 6. 참고 자료

- [Cython 공식 문서](https://cython.readthedocs.io/)
- [PyInstaller 공식 문서](https://pyinstaller.org/)
- [Python 코드 보호 방법](https://docs.python.org/3/faq/programming.html#how-can-i-protect-python-code)

