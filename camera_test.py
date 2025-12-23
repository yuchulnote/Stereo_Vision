# -*- coding: utf-8 -*-
"""
OpenCV 카메라 테스트 및 선택 유틸리티
"""

import cv2
import sys
import platform


def get_camera_names_windows():
    """
    Windows에서 모든 카메라 장치 이름을 DirectShow 인덱스 순서대로 가져오기
    
    Returns:
        list: 카메라 이름 리스트 (DirectShow 인덱스 순서)
    """
    device_names = []
    
    # 방법 1: pygrabber 라이브러리 사용 (가장 확실한 방법)
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        if devices:
            device_names = list(devices)
            return device_names
    except ImportError:
        pass
    except Exception:
        pass
    
    # 방법 2: PowerShell로 DirectShow 필터 이름 가져오기 (더 상세한 정보)
    if not device_names:
        try:
            import subprocess
            ps_script = '''
            Add-Type -TypeDefinition @"
            using System;
            using System.Runtime.InteropServices;
            using System.Collections.Generic;
            public class DirectShowHelper {
                [DllImport("ole32.dll")]
                public static extern int CoCreateInstance(
                    [MarshalAs(UnmanagedType.LPStruct)] Guid rclsid,
                    IntPtr pUnkOuter,
                    uint dwClsContext,
                    [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
                    out IntPtr ppv);
                
                public static List<string> GetVideoDevices() {
                    List<string> devices = new List<string>();
                    try {
                        Guid CLSID_SystemDeviceEnum = new Guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86");
                        Guid IID_ICreateDevEnum = new Guid("29840822-5B84-11D0-BD3B-00A0C911CE86");
                        IntPtr pDevEnum;
                        int hr = CoCreateInstance(CLSID_SystemDeviceEnum, IntPtr.Zero, 1, IID_ICreateDevEnum, out pDevEnum);
                        if (hr == 0) {
                            // 성공적으로 생성됨
                        }
                    } catch {}
                    return devices;
                }
            }
"@
            $devices = @()
            $regPath = "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\DirectShow\\VideoCaptureDevices"
            if (Test-Path $regPath) {
                Get-ChildItem $regPath | ForEach-Object {
                    $devices += $_.PSChildName
                }
            }
            $devices
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0 and result.stdout.strip():
                device_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
        except Exception:
            pass
    
    # 방법 3: Windows Registry에서 DirectShow VideoCaptureDevices 읽기
    if not device_names:
        try:
            import winreg
            
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\DirectShow\VideoCaptureDevices"
            
            try:
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                
                # 모든 서브키 열거 (인덱스 순서대로)
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        device_names.append(subkey_name)
                        i += 1
                    except OSError:
                        break
                
                winreg.CloseKey(key)
            except (FileNotFoundError, OSError):
                pass
            except Exception:
                pass
        except ImportError:
            pass
        except Exception:
            pass
    
    # 방법 4: PowerShell로 PnP 장치에서 카메라 이름 가져오기
    if not device_names:
        try:
            import subprocess
            ps_script = '''
            Get-PnpDevice -Class Camera | Where-Object {$_.Status -eq "OK"} | 
            Select-Object -ExpandProperty FriendlyName | 
            Where-Object {$_ -notmatch "printer|scanner|officejet|hp230|hp211|hp250|hp7720|hp8710"}
            '''
            
            result = subprocess.run(
                ['powershell', '-Command', ps_script],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if result.returncode == 0 and result.stdout.strip():
                device_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]
        except Exception:
            pass
    
    return device_names


def get_camera_name_by_index(index):
    """
    특정 인덱스의 카메라 이름을 직접 조회
    
    Args:
        index: 카메라 인덱스
        
    Returns:
        str: 카메라 이름 또는 None
    """
    # 방법 1: pygrabber 사용 (가장 확실한 방법, 설치 필요: pip install pygrabber)
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        devices = graph.get_input_devices()
        if devices and index < len(devices):
            return list(devices)[index]
    except ImportError:
        # pygrabber가 설치되지 않은 경우 무시
        pass
    except Exception:
        pass
    
    # 방법 2: PowerShell로 DirectShow 필터 이름 가져오기 (레지스트리 순서)
    try:
        import subprocess
        ps_script = f'''
        $regPath = "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\DirectShow\\VideoCaptureDevices"
        if (Test-Path $regPath) {{
            $devices = @()
            Get-ChildItem $regPath | ForEach-Object {{
                $devices += $_.PSChildName
            }}
            if ($devices.Count -gt {index}) {{
                $devices[{index}]
            }}
        }}
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip()
            if name and name.lower() not in ['printer', 'scanner', 'officejet']:
                return name
    except Exception:
        pass
    
    # 방법 3: WMI를 사용하여 카메라 이름 가져오기
    try:
        import subprocess
        ps_script = f'''
        $cameras = Get-WmiObject Win32_PnPEntity | Where-Object {{
            ($_.PNPClass -eq "Camera" -or $_.PNPClass -eq "Image") -and 
            $_.Status -eq "OK"
        }} | Select-Object -ExpandProperty Name | 
        Where-Object {{$_ -notmatch "printer|scanner|officejet|hp230|hp211|hp250|hp7720|hp8710"}}
        
        if ($cameras -and $cameras.Count -gt {index}) {{
            if ($cameras -is [array]) {{
                $cameras[{index}]
            }} else {{
                $cameras | Select-Object -Index {index}
            }}
        }}
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip()
            if name and name.lower() not in ['printer', 'scanner', 'officejet']:
                return name
    except Exception:
        pass
    
    # 방법 4: PnP 장치에서 특정 인덱스의 카메라 이름 가져오기
    try:
        import subprocess
        ps_script = f'''
        $cameras = Get-PnpDevice -Class Camera | Where-Object {{$_.Status -eq "OK"}} | 
                   Select-Object -ExpandProperty FriendlyName |
                   Where-Object {{$_ -notmatch "printer|scanner|officejet|hp230|hp211|hp250|hp7720|hp8710"}}
        if ($cameras -and $cameras.Count -gt {index}) {{
            if ($cameras -is [array]) {{
                $cameras[{index}]
            }} else {{
                $cameras | Select-Object -Index {index}
            }}
        }}
        '''
        
        result = subprocess.run(
            ['powershell', '-Command', ps_script],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            name = result.stdout.strip()
            if name and name.lower() not in ['printer', 'scanner', 'officejet']:
                return name
    except Exception:
        pass
    
    return None


def get_camera_info(index):
    """
    특정 인덱스의 카메라 정보 가져오기
    
    Args:
        index: 카메라 인덱스
        
    Returns:
        dict: 카메라 정보 (name, backend, resolution, fps 등)
    """
    info = {
        'index': index,
        'name': '알 수 없음',
        'backend': '알 수 없음',
        'available': False,
        'resolution': None,
        'fps': None
    }
    
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        info['available'] = True
        
        # 백엔드 정보 가져오기
        backend = cap.getBackendName()
        if backend:
            info['backend'] = backend
        
        # 해상도 정보 가져오기
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            info['resolution'] = f"{width}x{height}"
        
        # FPS 정보 가져오기
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            info['fps'] = f"{fps:.1f}"
        
        # Windows에서 카메라 이름 가져오기 (여러 방법 시도)
        if platform.system() == 'Windows':
            # 먼저 전체 리스트에서 찾기
            camera_names = get_camera_names_windows()
            if camera_names and index < len(camera_names):
                info['name'] = camera_names[index]
            else:
                # 전체 리스트에서 못 찾으면 인덱스로 직접 조회
                direct_name = get_camera_name_by_index(index)
                if direct_name:
                    info['name'] = direct_name
        
        cap.release()
    
    return info


def test_camera_index(index):
    """
    특정 인덱스의 카메라가 접근 가능한지 테스트
    
    Args:
        index: 카메라 인덱스
        
    Returns:
        bool: 카메라 접근 가능 여부
    """
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    return False


def get_available_cameras(max_index=10):
    """
    접근 가능한 모든 카메라 인덱스와 정보를 찾기
    
    Args:
        max_index: 테스트할 최대 인덱스 (기본값: 10)
        
    Returns:
        tuple: (available_cameras 리스트, camera_info_dict 딕셔너리)
    """
    available_cameras = []
    camera_info_dict = {}
    
    print("카메라 인덱스를 검색 중...")
    for i in range(max_index):
        info = get_camera_info(i)
        camera_info_dict[i] = info
        
        if info['available']:
            available_cameras.append(i)
            name_str = f" - {info['name']}" if info['name'] != '알 수 없음' else ""
            backend_str = f" ({info['backend']})" if info['backend'] != '알 수 없음' else ""
            print(f"  인덱스 {i}: 사용 가능{name_str}{backend_str}")
        else:
            print(f"  인덱스 {i}: 사용 불가")
    
    return available_cameras, camera_info_dict


def select_camera():
    """
    사용 가능한 카메라 목록을 보여주고 사용자가 선택하도록 함
    
    Returns:
        int: 선택된 카메라 인덱스, None if cancelled
    """
    print("\n" + "="*50)
    print("카메라 선택")
    print("="*50)
    
    # 접근 가능한 카메라 찾기
    available_cameras, camera_info_dict = get_available_cameras()
    
    if not available_cameras:
        print("\n사용 가능한 카메라를 찾을 수 없습니다.")
        return None
    
    print("\n" + "-"*70)
    print("사용 가능한 카메라:")
    print("-"*70)
    for idx, camera_idx in enumerate(available_cameras):
        info = camera_info_dict[camera_idx]
        name = info['name']
        backend = info['backend']
        resolution = info['resolution'] or "알 수 없음"
        fps = info['fps'] or "알 수 없음"
        
        print(f"  [{idx}] 인덱스: {camera_idx}")
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
            
            if 0 <= choice_num < len(available_cameras):
                selected_index = available_cameras[choice_num]
                selected_info = camera_info_dict[selected_index]
                print(f"\n카메라 인덱스 {selected_index}를 선택했습니다.")
                print(f"  이름: {selected_info['name']}")
                print(f"  백엔드: {selected_info['backend']}")
                return selected_index
            else:
                print(f"잘못된 입력입니다. 0-{len(available_cameras) - 1} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("숫자를 입력해주세요. (취소: q 또는 c)")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            return None


def open_camera_stream(camera_index):
    """
    선택된 카메라로 영상 스트림 열기
    
    Args:
        camera_index: 카메라 인덱스
        
    Returns:
        cv2.VideoCapture: 카메라 객체, None if failed
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"카메라 인덱스 {camera_index}를 열 수 없습니다.")
        return None
    
    # 카메라 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print(f"\n카메라 인덱스 {camera_index}가 성공적으로 열렸습니다.")
    print("영상 스트림을 시작합니다. 'q' 키를 눌러 종료하세요.\n")
    
    return cap


def main():
    """메인 함수"""
    # pygrabber 설치 안내 (선택사항)
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("\n참고: 더 정확한 카메라 이름을 보려면 'pygrabber' 라이브러리를 설치하세요.")
        print("      설치 방법: pip install pygrabber\n")
    
    # 카메라 선택
    camera_index = select_camera()
    
    if camera_index is None:
        print("프로그램을 종료합니다.")
        return
    
    # 카메라 열기
    cap = open_camera_stream(camera_index)
    
    if cap is None:
        print("카메라를 열 수 없습니다.")
        return
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # 프레임에 정보 표시
            cv2.putText(frame, f"Camera Index: {camera_index}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 영상 표시
            cv2.imshow('Camera Test', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        print("카메라가 닫혔습니다.")


if __name__ == "__main__":
    main()

