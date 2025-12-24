# -*- coding: utf-8 -*-
"""
스테레오 비전 메인 프로세스
카메라 Producer 프로세스들을 관리하고 뷰어를 실행
"""

import multiprocessing as mp
import time
import atexit
import sys
from pathlib import Path
from typing import Optional, Dict

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.shared_buffer import SharedRingBuffer
from core.camera_producer import camera_producer_process
from core.stereo_viewer import StereoViewer
from core.pose_inference import pose_inference_process
from utils.camera_config import load_config
from utils.logger import get_logger
import camera_test  # 카메라 선택 함수 사용


class StereoVisionSystem:
    """스테레오 비전 시스템 메인 클래스"""
    
    def __init__(self, config_path: str = "config.yaml", camera_0_index: Optional[int] = None, camera_1_index: Optional[int] = None):
        """
        Args:
            config_path: 설정 파일 경로
            camera_0_index: 카메라 0 인덱스 (None이면 선택 화면 표시)
            camera_1_index: 카메라 1 인덱스 (None이면 선택 화면 표시)
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        
        self.logger = get_logger(self.config.get('logging', {}))
        
        # 카메라 선택
        if camera_0_index is None or camera_1_index is None:
            print("\n" + "="*70)
            print("스테레오 비전 카메라 선택")
            print("="*70)
            
            # 사용 가능한 카메라 목록 가져오기
            available_cameras, camera_info_dict = camera_test.get_available_cameras()
            
            if len(available_cameras) < 2:
                raise ValueError(f"스테레오 비전을 위해서는 최소 2개의 카메라가 필요합니다. (현재: {len(available_cameras)}개)")
            
            if camera_0_index is None:
                print("\n[카메라 0 선택]")
                camera_0_index = self._select_camera_interactive(available_cameras, camera_info_dict, None)
                if camera_0_index is None:
                    raise ValueError("카메라 0 선택이 취소되었습니다.")
            
            if camera_1_index is None:
                print("\n[카메라 1 선택]")
                # 이미 선택된 카메라는 제외
                camera_1_index = self._select_camera_interactive(available_cameras, camera_info_dict, camera_0_index)
                if camera_1_index is None:
                    raise ValueError("카메라 1 선택이 취소되었습니다.")
            
            # 같은 카메라 선택 방지
            if camera_0_index == camera_1_index:
                raise ValueError(f"같은 카메라를 선택할 수 없습니다. (인덱스: {camera_0_index})")
            
            # 선택된 카메라 정보 출력
            print("\n" + "="*70)
            print("선택된 카메라 정보")
            print("="*70)
            info_0 = camera_info_dict.get(camera_0_index, {})
            info_1 = camera_info_dict.get(camera_1_index, {})
            print(f"카메라 0: 인덱스 {camera_0_index} - {info_0.get('name', '알 수 없음')}")
            print(f"카메라 1: 인덱스 {camera_1_index} - {info_1.get('name', '알 수 없음')}")
            print("="*70)
        
        self.camera_0_index = camera_0_index
        self.camera_1_index = camera_1_index
        
        # 카메라 설정 (선택된 인덱스에 맞춰 config에서 찾기)
        camera_0_config = self.config.get('cameras', {}).get('camera_0', {})
        camera_1_config = self.config.get('cameras', {}).get('camera_1', {})
        
        # 두 카메라의 실제 해상도 확인 및 동기화
        self.width, self.height = self._verify_and_sync_resolutions(
            camera_0_index, camera_1_index, camera_0_config, camera_1_config
        )
        self.channels = 3  # RGB
        
        # 버퍼 설정
        self.buffer_size = 3  # 원형 버퍼 크기
        
        # 공유 메모리 버퍼
        self.buffer_0: Optional[SharedRingBuffer] = None
        self.buffer_1: Optional[SharedRingBuffer] = None
        
        # Producer 프로세스
        self.producer_0: Optional[mp.Process] = None
        self.producer_1: Optional[mp.Process] = None
        
        # 포즈 추론 프로세스
        self.pose_inference_0: Optional[mp.Process] = None
        self.pose_inference_1: Optional[mp.Process] = None
        
        # 포즈 추론 결과 큐
        self.pose_queue_0: Optional[mp.Queue] = None
        self.pose_queue_1: Optional[mp.Queue] = None
        
        # 뷰어
        self.viewer: Optional[StereoViewer] = None
        
        # 종료 핸들러 등록 (Step 21)
        atexit.register(self.cleanup)
        
        # 프로세스 종료 시그널 핸들러
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _select_camera_interactive(self, available_cameras, camera_info_dict, exclude_index: Optional[int] = None):
        """
        대화형 카메라 선택
        
        Args:
            available_cameras: 사용 가능한 카메라 인덱스 리스트
            camera_info_dict: 카메라 정보 딕셔너리
            exclude_index: 제외할 카메라 인덱스 (None이면 제외 없음)
            
        Returns:
            선택된 카메라 인덱스 또는 None
        """
        # 제외할 카메라 필터링
        filtered_cameras = [idx for idx in available_cameras if idx != exclude_index]
        
        if not filtered_cameras:
            print("선택 가능한 카메라가 없습니다.")
            return None
        
        print("\n" + "-"*70)
        print("사용 가능한 카메라:")
        print("-"*70)
        for idx, camera_idx in enumerate(filtered_cameras):
            info = camera_info_dict.get(camera_idx, {})
            name = info.get('name', '알 수 없음')
            backend = info.get('backend', '알 수 없음')
            resolution = info.get('resolution', '알 수 없음')
            fps = info.get('fps', '알 수 없음')
            
            marker = " [이미 선택됨]" if camera_idx == exclude_index else ""
            print(f"  [{idx}] 인덱스: {camera_idx}{marker}")
            print(f"      이름: {name}")
            print(f"      백엔드: {backend}")
            print(f"      해상도: {resolution}")
            print(f"      FPS: {fps}")
            print()
        print("  취소하려면 'q' 또는 'c' 입력")
        print("-"*70)
        
        # 사용자 입력 받기
        while True:
            try:
                choice = input("\n카메라를 선택하세요 (번호 입력, 취소: q/c): ").strip().lower()
                
                # 취소 처리
                if choice == 'q' or choice == 'c':
                    print("취소되었습니다.")
                    return None
                
                choice_num = int(choice)
                
                if 0 <= choice_num < len(filtered_cameras):
                    selected_index = filtered_cameras[choice_num]
                    selected_info = camera_info_dict.get(selected_index, {})
                    print(f"\n카메라 인덱스 {selected_index}를 선택했습니다.")
                    print(f"  이름: {selected_info.get('name', '알 수 없음')}")
                    print(f"  백엔드: {selected_info.get('backend', '알 수 없음')}")
                    return selected_index
                else:
                    print(f"잘못된 입력입니다. 0-{len(filtered_cameras) - 1} 사이의 숫자를 입력하세요.")
            except ValueError:
                print("숫자를 입력해주세요. (취소: q 또는 c)")
            except KeyboardInterrupt:
                print("\n취소되었습니다.")
                return None
    
    def _verify_and_sync_resolutions(
        self,
        camera_0_index: int,
        camera_1_index: int,
        camera_0_config: Dict,
        camera_1_config: Dict
    ) -> tuple:
        """
        두 카메라의 해상도를 확인하고 동일하게 맞춤
        
        Args:
            camera_0_index: 카메라 0 인덱스
            camera_1_index: 카메라 1 인덱스
            camera_0_config: 카메라 0 설정
            camera_1_config: 카메라 1 설정
            
        Returns:
            (width, height) 튜플
        """
        import cv2
        from utils.camera_config import apply_camera_settings
        
        # config에서 해상도 가져오기 (기본값 사용)
        target_width = camera_0_config.get('width', 640)
        target_height = camera_0_config.get('height', 480)
        
        # 두 카메라를 열어서 실제 해상도 확인
        cap_0 = cv2.VideoCapture(camera_0_index)
        cap_1 = cv2.VideoCapture(camera_1_index)
        
        try:
            if not cap_0.isOpened():
                raise ValueError(f"카메라 {camera_0_index}를 열 수 없습니다.")
            if not cap_1.isOpened():
                raise ValueError(f"카메라 {camera_1_index}를 열 수 없습니다.")
            
            # 설정 적용
            apply_camera_settings(cap_0, camera_0_config)
            apply_camera_settings(cap_1, camera_1_config)
            
            # 실제 해상도 확인
            actual_width_0 = int(cap_0.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height_0 = int(cap_0.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_width_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(
                f"카메라 해상도 확인 - "
                f"카메라 {camera_0_index}: {actual_width_0}x{actual_height_0}, "
                f"카메라 {camera_1_index}: {actual_width_1}x{actual_height_1}"
            )
            
            # 해상도가 다르면 경고 및 동기화
            if (actual_width_0, actual_height_0) != (actual_width_1, actual_height_1):
                self.logger.warning(
                    f"두 카메라의 해상도가 다릅니다! "
                    f"카메라 {camera_0_index}: {actual_width_0}x{actual_height_0}, "
                    f"카메라 {camera_1_index}: {actual_width_1}x{actual_height_1}"
                )
                
                # 더 낮은 해상도로 통일 (성능 고려)
                if actual_width_0 * actual_height_0 < actual_width_1 * actual_height_1:
                    target_width, target_height = actual_width_0, actual_height_0
                    self.logger.info(
                        f"카메라 {camera_1_index}의 해상도를 "
                        f"{target_width}x{target_height}로 변경합니다."
                    )
                    cap_1.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap_1.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                else:
                    target_width, target_height = actual_width_1, actual_height_1
                    self.logger.info(
                        f"카메라 {camera_0_index}의 해상도를 "
                        f"{target_width}x{target_height}로 변경합니다."
                    )
                    cap_0.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
                    cap_0.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
                
                # 변경 후 다시 확인
                time.sleep(0.1)  # 설정 적용 대기
                final_width_0 = int(cap_0.get(cv2.CAP_PROP_FRAME_WIDTH))
                final_height_0 = int(cap_0.get(cv2.CAP_PROP_FRAME_HEIGHT))
                final_width_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
                final_height_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if (final_width_0, final_height_0) != (final_width_1, final_height_1):
                    self.logger.error(
                        f"해상도 동기화 실패! "
                        f"카메라 {camera_0_index}: {final_width_0}x{final_height_0}, "
                        f"카메라 {camera_1_index}: {final_width_1}x{final_height_1}"
                    )
                    # 더 낮은 해상도로 강제 통일
                    target_width = min(final_width_0, final_width_1)
                    target_height = min(final_height_0, final_height_1)
                    self.logger.warning(
                        f"최소 해상도로 통일: {target_width}x{target_height}"
                    )
                else:
                    target_width, target_height = final_width_0, final_height_0
                    self.logger.info(
                        f"해상도 동기화 완료: {target_width}x{target_height}"
                    )
            else:
                # 이미 동일한 해상도
                target_width, target_height = actual_width_0, actual_height_0
                self.logger.info(
                    f"두 카메라의 해상도가 동일합니다: {target_width}x{target_height}"
                )
            
            return target_width, target_height
            
        finally:
            cap_0.release()
            cap_1.release()
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"시그널 {signum} 수신, 종료 중...")
        self.cleanup()
        sys.exit(0)
    
    def initialize_buffers(self):
        """공유 메모리 버퍼 초기화"""
        try:
            # 버퍼 0 생성
            self.buffer_0 = SharedRingBuffer(
                buffer_size=self.buffer_size,
                width=self.width,
                height=self.height,
                channels=self.channels
            )
            
            # 버퍼 1 생성
            self.buffer_1 = SharedRingBuffer(
                buffer_size=self.buffer_size,
                width=self.width,
                height=self.height,
                channels=self.channels
            )
            
            self.logger.info(
                f"공유 메모리 버퍼 초기화 완료: "
                f"{self.width}x{self.height}, 버퍼 크기={self.buffer_size}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"버퍼 초기화 실패: {e}")
            return False
    
    def start_producers(self):
        """Producer 프로세스 시작"""
        try:
            # Producer 0 시작 (카메라 0)
            self.producer_0 = mp.Process(
                target=camera_producer_process,
                args=(
                    self.camera_0_index,
                    self.buffer_0,  # 객체 자체 전달
                    self.config_path,
                    1.0  # reconnect_interval
                ),
                name=f"CameraProducer-{self.camera_0_index}"
            )
            self.producer_0.start()
            
            # Producer 1 시작 (카메라 1)
            self.producer_1 = mp.Process(
                target=camera_producer_process,
                args=(
                    self.camera_1_index,
                    self.buffer_1,  # 객체 자체 전달
                    self.config_path,
                    1.0  # reconnect_interval
                ),
                name=f"CameraProducer-{self.camera_1_index}"
            )
            self.producer_1.start()
            
            self.logger.info("카메라 Producer 프로세스 시작 완료")
            
            # 프로세스 시작 대기
            time.sleep(1.0)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Producer 프로세스 시작 실패: {e}")
            return False
    
    def start_pose_inference(self):
        """포즈 추론 프로세스 시작"""
        try:
            # 포즈 추론 사용 여부 확인
            use_pose_estimation = self.config.get('pose_estimation', {}).get('enabled', True)
            if not use_pose_estimation:
                self.logger.info("포즈 추론이 비활성화되어 있습니다.")
                return True
            
            # 포즈 추론 결과 큐 생성
            self.pose_queue_0 = mp.Queue(maxsize=10)
            self.pose_queue_1 = mp.Queue(maxsize=10)
            
            # 포즈 추론 프로세스 0 시작
            self.pose_inference_0 = mp.Process(
                target=pose_inference_process,
                args=(
                    self.camera_0_index,
                    self.buffer_0,
                    self.pose_queue_0,
                    self.config_path
                ),
                name=f"PoseInference-{self.camera_0_index}"
            )
            self.pose_inference_0.start()
            
            # 포즈 추론 프로세스 1 시작
            self.pose_inference_1 = mp.Process(
                target=pose_inference_process,
                args=(
                    self.camera_1_index,
                    self.buffer_1,
                    self.pose_queue_1,
                    self.config_path
                ),
                name=f"PoseInference-{self.camera_1_index}"
            )
            self.pose_inference_1.start()
            
            self.logger.info("포즈 추론 프로세스 시작 완료")
            
            # 프로세스 시작 대기
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            self.logger.error(f"포즈 추론 프로세스 시작 실패: {e}")
            return False
    
    def start_viewer(self):
        """뷰어 시작"""
        try:
            self.viewer = StereoViewer(
                buffer_0=self.buffer_0,
                buffer_1=self.buffer_1,
                config=self.config,
                config_path=self.config_path,
                pose_queue_0=self.pose_queue_0,
                pose_queue_1=self.pose_queue_1,
                pose_inference_0=self.pose_inference_0,
                pose_inference_1=self.pose_inference_1
            )
            
            # 뷰어 실행 (블로킹)
            self.viewer.run()
            
        except Exception as e:
            self.logger.error(f"뷰어 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """시스템 실행"""
        self.logger.info("스테레오 비전 시스템 시작")
        
        # 버퍼 초기화
        if not self.initialize_buffers():
            return
        
        # Producer 프로세스 시작
        if not self.start_producers():
            self.cleanup()
            return
        
        # 포즈 추론 프로세스 시작
        if not self.start_pose_inference():
            self.logger.warning("포즈 추론 프로세스 시작 실패, 포즈 추론 없이 계속 진행")
        
        try:
            # 뷰어 시작
            self.start_viewer()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리 작업 (Step 21: 좀비 프로세스 방지)"""
        self.logger.info("시스템 정리 시작")
        
        # 뷰어 정리
        if self.viewer is not None:
            try:
                self.viewer.cleanup()
            except Exception as e:
                self.logger.error(f"뷰어 정리 중 오류: {e}")
            finally:
                self.viewer = None
        
        # 포즈 추론 프로세스 종료
        if self.pose_inference_0 is not None:
            if self.pose_inference_0.is_alive():
                self.pose_inference_0.terminate()
                self.pose_inference_0.join(timeout=5.0)
                if self.pose_inference_0.is_alive():
                    self.logger.warning("Pose Inference 0 강제 종료")
                    self.pose_inference_0.kill()
                    self.pose_inference_0.join()
            self.pose_inference_0 = None
        
        if self.pose_inference_1 is not None:
            if self.pose_inference_1.is_alive():
                self.pose_inference_1.terminate()
                self.pose_inference_1.join(timeout=5.0)
                if self.pose_inference_1.is_alive():
                    self.logger.warning("Pose Inference 1 강제 종료")
                    self.pose_inference_1.kill()
                    self.pose_inference_1.join()
            self.pose_inference_1 = None
        
        # Producer 프로세스 종료
        if self.producer_0 is not None:
            if self.producer_0.is_alive():
                self.producer_0.terminate()
                self.producer_0.join(timeout=5.0)
                if self.producer_0.is_alive():
                    self.logger.warning("Producer 0 강제 종료")
                    self.producer_0.kill()
                    self.producer_0.join()
            self.producer_0 = None
        
        if self.producer_1 is not None:
            if self.producer_1.is_alive():
                self.producer_1.terminate()
                self.producer_1.join(timeout=5.0)
                if self.producer_1.is_alive():
                    self.logger.warning("Producer 1 강제 종료")
                    self.producer_1.kill()
                    self.producer_1.join()
            self.producer_1 = None
        
        # 버퍼 정리
        if self.buffer_0 is not None:
            self.buffer_0.close()
            self.buffer_0 = None
        
        if self.buffer_1 is not None:
            self.buffer_1.close()
            self.buffer_1 = None
        
        self.logger.info("시스템 정리 완료")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='스테레오 비전 시스템')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='설정 파일 경로'
    )
    parser.add_argument(
        '--camera0',
        type=int,
        default=None,
        help='카메라 0 인덱스 (지정하지 않으면 선택 화면 표시)'
    )
    parser.add_argument(
        '--camera1',
        type=int,
        default=None,
        help='카메라 1 인덱스 (지정하지 않으면 선택 화면 표시)'
    )
    
    args = parser.parse_args()
    
    # multiprocessing 시작 방법 설정
    mp.set_start_method('spawn', force=True)
    
    try:
        # 시스템 생성 및 실행
        system = StereoVisionSystem(
            config_path=args.config,
            camera_0_index=args.camera0,
            camera_1_index=args.camera1
        )
        system.run()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\n오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

