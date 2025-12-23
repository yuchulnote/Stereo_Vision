# -*- coding: utf-8 -*-
"""
Cython 컴파일을 위한 setup.py
핵심 알고리즘 모듈을 Cython으로 컴파일하여 성능 향상 및 코드 보호
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Cython 컴파일 대상 모듈 목록
# 주의: 실제 파일이 존재할 때만 컴파일하도록 설정
extensions = []

# NumPy include 디렉토리
numpy_include = numpy.get_include()

# 컴파일 옵션
compile_args = ['-O3', '-march=native'] if os.name != 'nt' else ['/O2']
link_args = []

# Cython 컴파일러 지시문
compiler_directives = {
    'language_level': "3",
    'boundscheck': False,  # 경계 검사 비활성화 (성능 향상)
    'wraparound': False,    # 음수 인덱스 비활성화
    'cdivision': True,      # C 스타일 나눗셈
    'initializedcheck': False,  # 초기화 검사 비활성화
    'nonecheck': False,     # None 검사 비활성화
}

# 확장 모듈 정의 (실제 파일이 생성되면 활성화)
# 예시:
# extensions.append(
#     Extension(
#         "core.stereo_vision",
#         ["core/stereo_vision.pyx"],
#         include_dirs=[numpy_include],
#         extra_compile_args=compile_args,
#         extra_link_args=link_args,
#         language="c++",  # C++ 사용 시
#     )
# )

setup(
    name="stereo_vision",
    version="0.1.0",
    description="스테레오 비전 프로젝트",
    author="Your Name",
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        build_dir="build",
    ),
    zip_safe=False,
    install_requires=[
        'numpy',
        'opencv-python',
        'cython',
    ],
)

# 사용 방법:
# 1. .py 파일을 .pyx로 변환하거나 직접 .pyx 파일 작성
# 2. 위의 extensions 리스트에 모듈 추가
# 3. python setup.py build_ext --inplace 실행
# 4. 컴파일된 .so (Linux) 또는 .pyd (Windows) 파일이 생성됨

