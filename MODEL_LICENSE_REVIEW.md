# 객체 인식/포즈 추정 모델 라이선스 검토

## 검토 일자
2024년

## 검토 목적
스테레오 비전 프로젝트에서 사용할 객체 인식 및 포즈 추정 모델 선정을 위한 라이선스 검토

## 검토 대상 모델

### 1. YOLOv8 (Ultralytics)
- **라이선스**: AGPL-3.0
- **상업적 사용 제한**: AGPL은 상업적 배포 시 전체 소스 코드 공개 의무
- **결정**: ❌ **제외** (상용 배포 고려 시 부적합)

### 2. RTMPose (MMPose)
- **라이선스**: Apache 2.0
- **상업적 사용**: ✅ 자유롭게 사용 가능 (소스 코드 공개 의무 없음)
- **특징**:
  - 정확한 포즈 추정
  - 실시간 성능
  - 다양한 모델 변형 제공
- **결정**: ✅ **선정 후보 1**

### 3. MediaPipe
- **라이선스**: Apache 2.0
- **상업적 사용**: ✅ 자유롭게 사용 가능
- **특징**:
  - Google 개발
  - 경량화 및 최적화
  - 실시간 포즈 추정
  - 크로스 플랫폼 지원
- **결정**: ✅ **선정 후보 2**

## 최종 결정

### 선정 모델: **MediaPipe** 또는 **RTMPose**

#### MediaPipe 선택 시 장점:
- Google의 안정적인 지원
- 경량화된 모델
- 쉬운 통합
- 다양한 플랫폼 지원

#### RTMPose 선택 시 장점:
- 더 높은 정확도
- 다양한 모델 크기 옵션
- 연구용으로 널리 사용됨

## 권장사항

프로젝트 초기에는 **MediaPipe**를 사용하여 빠른 프로토타이핑을 진행하고, 
정확도가 더 필요한 경우 **RTMPose**로 전환하는 것을 권장합니다.

## 라이선스 준수 사항

- Apache 2.0 라이선스 하에서 사용 시:
  - 라이선스 고지 유지
  - 저작권 표시 유지
  - 변경 사항 명시 (있는 경우)
  - NOTICE 파일 포함 (있는 경우)

## 참고 자료

- [MediaPipe License](https://github.com/google/mediapipe/blob/master/LICENSE)
- [RTMPose License](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)
- [YOLOv8 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

