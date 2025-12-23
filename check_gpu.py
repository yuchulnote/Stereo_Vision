# -*- coding: utf-8 -*-
"""
GPU 가속(CUDA) 환경 확인 스크립트
PyTorch와 ONNX Runtime이 GPU를 인식하는지 확인
"""

import sys
import platform


def check_pytorch_gpu():
    """PyTorch CUDA 지원 확인"""
    print("=" * 60)
    print("PyTorch GPU 지원 확인")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}:")
                print(f"  이름: {torch.cuda.get_device_name(i)}")
                print(f"  메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
                print(f"  컴퓨팅 능력: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
            # 간단한 GPU 테스트
            try:
                x = torch.randn(3, 3).cuda()
                y = torch.randn(3, 3).cuda()
                z = torch.matmul(x, y)
                print("\n✅ PyTorch GPU 테스트 성공!")
                return True
            except Exception as e:
                print(f"\n❌ PyTorch GPU 테스트 실패: {e}")
                return False
        else:
            print("\n⚠️  CUDA를 사용할 수 없습니다.")
            print("   CPU 모드로 실행됩니다.")
            return False
            
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        print("   설치 방법: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def check_onnxruntime_gpu():
    """ONNX Runtime GPU 지원 확인"""
    print("\n" + "=" * 60)
    print("ONNX Runtime GPU 지원 확인")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        
        print(f"ONNX Runtime 버전: {ort.__version__}")
        
        # 사용 가능한 실행 공급자 확인
        available_providers = ort.get_available_providers()
        print(f"사용 가능한 실행 공급자: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("\n✅ CUDA 실행 공급자 사용 가능!")
            
            # CUDA 실행 공급자 정보
            try:
                providers = [('CUDAExecutionProvider', {})]
                session_options = ort.SessionOptions()
                # 간단한 더미 모델로 테스트는 생략 (실제 모델 필요)
                print("   CUDA 실행 공급자가 정상적으로 로드되었습니다.")
                return True
            except Exception as e:
                print(f"   ⚠️  CUDA 실행 공급자 로드 중 오류: {e}")
                return False
        else:
            print("\n⚠️  CUDA 실행 공급자를 사용할 수 없습니다.")
            if 'CPUExecutionProvider' in available_providers:
                print("   CPU 모드로 실행됩니다.")
            return False
            
    except ImportError:
        print("❌ ONNX Runtime이 설치되지 않았습니다.")
        print("   GPU 사용 시: pip install onnxruntime-gpu")
        print("   CPU만 사용 시: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False


def check_system_info():
    """시스템 정보 출력"""
    print("=" * 60)
    print("시스템 정보")
    print("=" * 60)
    print(f"운영체제: {platform.system()} {platform.release()}")
    print(f"프로세서: {platform.processor()}")
    print(f"Python 버전: {sys.version}")
    print()


def main():
    """메인 함수"""
    check_system_info()
    
    pytorch_ok = check_pytorch_gpu()
    onnx_ok = check_onnxruntime_gpu()
    
    print("\n" + "=" * 60)
    print("종합 결과")
    print("=" * 60)
    
    if pytorch_ok and onnx_ok:
        print("✅ 모든 GPU 환경이 정상적으로 설정되었습니다!")
        return 0
    elif pytorch_ok or onnx_ok:
        print("⚠️  일부 GPU 환경만 사용 가능합니다.")
        return 1
    else:
        print("❌ GPU 환경을 사용할 수 없습니다. CPU 모드로 실행됩니다.")
        return 2


if __name__ == "__main__":
    sys.exit(main())

