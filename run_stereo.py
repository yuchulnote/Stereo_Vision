# -*- coding: utf-8 -*-
"""
스테레오 비전 시스템 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.stereo_main import main

if __name__ == "__main__":
    main()

