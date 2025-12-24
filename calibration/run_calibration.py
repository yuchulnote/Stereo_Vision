# -*- coding: utf-8 -*-
"""
수집된 이미지로 스테레오 캘리브레이션 수행
"""

import cv2
import numpy as np
import glob
import os
import yaml
from pathlib import Path
import argparse

def run_intrinsic_calibration(
    image_dir: str = "data/calibration_images",
    output_file: str = "intrinsic_calibration.yaml",
    square_size: float = 30.0,
    chessboard_size: tuple = (8, 5),  # (가로 내부 코너 수, 세로 내부 코너 수)
    camera_side: str = "left"  # "left" or "right"
):
    """
    개별 카메라 Intrinsic 캘리브레이션 실행
    공장 출하 단계 혹은 렌즈 변경 시 1회 수행 (변하지 않음)
    
    Args:
        image_dir: 이미지가 저장된 디렉토리
        output_file: 결과 저장 파일 경로
        square_size: 체스보드 사각형 한 변의 길이 (mm)
        chessboard_size: (cols, rows) 내부 코너 개수
        camera_side: "left" 또는 "right"
    """
    print(f"Intrinsic 캘리브레이션 시작 (카메라: {camera_side})...")
    print(f"  - 이미지 경로: {image_dir}")
    print(f"  - 체스보드 패턴: {chessboard_size}")
    print(f"  - 사각형 크기: {square_size} mm")
    
    # 3D 월드 좌표점 생성
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 코너점 저장 리스트
    objpoints = []
    imgpoints = []
    
    # 이미지 파일 목록 가져오기
    if camera_side == "left":
        image_files = sorted(glob.glob(os.path.join(image_dir, "left_*.jpg")))
    else:
        image_files = sorted(glob.glob(os.path.join(image_dir, "right_*.jpg")))
    
    if len(image_files) < 10:
        print(f"[경고] 이미지가 너무 적습니다 ({len(image_files)}장). 최소 10장 이상 권장합니다.")
    
    valid_count = 0
    image_size = None
    
    print("\n[이미지 처리 중...]")
    for img_path in image_files:
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 찾기
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, flags)
        
        if ret:
            # 정밀도 향상 (Subpixel)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_count += 1
            print(f"  - OK: {os.path.basename(img_path)}")
        else:
            print(f"  - Skip (감지 실패): {os.path.basename(img_path)}")
            
    print(f"\n총 {len(image_files)}장 중 {valid_count}장 유효함.")
    
    if valid_count < 5:
        print("[실패] 유효한 이미지가 너무 적어 캘리브레이션을 수행할 수 없습니다.")
        return False
        
    # Intrinsic 캘리브레이션 수행
    print("\n[Intrinsic 캘리브레이션 수행 중...]")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    print(f"  - Reprojection Error: {ret:.4f}")
    
    # 결과 저장
    data = {
        'camera_matrix': mtx.tolist(),
        'dist_coeffs': dist.tolist(),
        'image_size': image_size,
        'reprojection_error': ret,
        'camera_side': camera_side,
        'square_size': square_size,
        'chessboard_size': chessboard_size
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(data, f)
        
    print(f"\n[성공] Intrinsic 캘리브레이션 결과가 {output_file}에 저장되었습니다.")
    return True


def run_extrinsic_calibration(
    image_dir: str = "data/calibration_images",
    intrinsic_left_file: str = "intrinsic_calibration_left.yaml",
    intrinsic_right_file: str = "intrinsic_calibration_right.yaml",
    output_file: str = "extrinsic_calibration.yaml",
    square_size: float = 30.0,
    chessboard_size: tuple = (8, 5)  # (가로 내부 코너 수, 세로 내부 코너 수)
):
    """
    스테레오 Extrinsic 캘리브레이션 실행
    카메라 설치 시마다 수행 (카메라 간 상대 위치/자세 계산)
    
    Args:
        image_dir: 이미지가 저장된 디렉토리
        intrinsic_left_file: 왼쪽 카메라 Intrinsic 캘리브레이션 파일 경로
        intrinsic_right_file: 오른쪽 카메라 Intrinsic 캘리브레이션 파일 경로
        output_file: 결과 저장 파일 경로
        square_size: 체스보드 사각형 한 변의 길이 (mm)
        chessboard_size: (cols, rows) 내부 코너 개수
    """
    print(f"Extrinsic 캘리브레이션 시작...")
    print(f"  - 이미지 경로: {image_dir}")
    print(f"  - Intrinsic 파일 (왼쪽): {intrinsic_left_file}")
    print(f"  - Intrinsic 파일 (오른쪽): {intrinsic_right_file}")
    print(f"  - 체스보드 패턴: {chessboard_size}")
    print(f"  - 사각형 크기: {square_size} mm")
    
    # Intrinsic 캘리브레이션 결과 로드
    try:
        with open(intrinsic_left_file, 'r') as f:
            if hasattr(yaml, 'FullLoader'):
                intrinsic_left = yaml.load(f, Loader=yaml.FullLoader)
            else:
                intrinsic_left = yaml.safe_load(f)
        
        with open(intrinsic_right_file, 'r') as f:
            if hasattr(yaml, 'FullLoader'):
                intrinsic_right = yaml.load(f, Loader=yaml.FullLoader)
            else:
                intrinsic_right = yaml.safe_load(f)
    except Exception as e:
        print(f"[오류] Intrinsic 캘리브레이션 파일 로드 실패: {e}")
        return False
    
    mtx_l = np.array(intrinsic_left['camera_matrix'])
    dist_l = np.array(intrinsic_left['dist_coeffs'])
    mtx_r = np.array(intrinsic_right['camera_matrix'])
    dist_r = np.array(intrinsic_right['dist_coeffs'])
    image_size = tuple(intrinsic_left['image_size'])
    
    # 3D 월드 좌표점 생성
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 코너점 저장 리스트
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []
    
    # 이미지 파일 목록 가져오기
    left_images = sorted(glob.glob(os.path.join(image_dir, "left_*.jpg")))
    right_images = sorted(glob.glob(os.path.join(image_dir, "right_*.jpg")))
    
    if len(left_images) != len(right_images):
        print(f"[오류] 왼쪽({len(left_images)})과 오른쪽({len(right_images)}) 이미지 개수가 다릅니다.")
        return False
        
    if len(left_images) < 10:
        print(f"[경고] 이미지가 너무 적습니다 ({len(left_images)}장). 최소 10장 이상 권장합니다.")
    
    valid_pairs = 0
    
    print("\n[이미지 처리 중...]")
    for left_img_path, right_img_path in zip(left_images, right_images):
        img_l = cv2.imread(left_img_path)
        img_r = cv2.imread(right_img_path)
        
        if img_l is None or img_r is None:
            continue
            
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        
        # 체스보드 찾기
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, flags)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, flags)
        
        if ret_l and ret_r:
            # 정밀도 향상 (Subpixel)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_l.append(corners_l)
            imgpoints_r.append(corners_r)
            valid_pairs += 1
            print(f"  - OK: {os.path.basename(left_img_path)}")
        else:
            print(f"  - Skip (감지 실패): {os.path.basename(left_img_path)}")
            
    print(f"\n총 {len(left_images)}장 중 {valid_pairs}장 유효함.")
    
    if valid_pairs < 5:
        print("[실패] 유효한 이미지가 너무 적어 캘리브레이션을 수행할 수 없습니다.")
        return False
        
    # Extrinsic 캘리브레이션 수행 (Intrinsic 고정)
    print("\n[Extrinsic 캘리브레이션 수행 중...]")
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret_stereo, mtx_l_new, dist_l_new, mtx_r_new, dist_r_new, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l,
        mtx_r, dist_r,
        image_size,
        criteria=criteria,
        flags=flags
    )
    
    print(f"  - Stereo Reprojection Error: {ret_stereo:.4f}")
    print(f"  - Translation Vector (T):\n{T}")
    
    # 결과 저장
    data = {
        'camera_matrix_left': mtx_l.tolist(),
        'dist_coeffs_left': dist_l.tolist(),
        'camera_matrix_right': mtx_r.tolist(),
        'dist_coeffs_right': dist_r.tolist(),
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        'image_size': image_size,
        'reprojection_error': ret_stereo
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(data, f)
        
    print(f"\n[성공] Extrinsic 캘리브레이션 결과가 {output_file}에 저장되었습니다.")
    return True


def run_calibration(
    image_dir: str = "data/calibration_images",
    output_file: str = "calibration_result.yaml",
    square_size: float = 30.0,
    chessboard_size: tuple = (8, 5),  # (가로 내부 코너 수, 세로 내부 코너 수)
    mode: str = "combined"  # "combined", "intrinsic", "extrinsic"
):
    """
    스테레오 캘리브레이션 실행 (통합 또는 분리 모드)
    
    Args:
        image_dir: 이미지가 저장된 디렉토리
        output_file: 결과 저장 파일 경로
        square_size: 체스보드 사각형 한 변의 길이 (mm)
        chessboard_size: (cols, rows) 내부 코너 개수
        mode: "combined" (기존 방식), "intrinsic" (Intrinsic만), "extrinsic" (Extrinsic만)
    """
    if mode == "combined":
        # 기존 방식: Intrinsic과 Extrinsic을 한 번에 계산
        print(f"캘리브레이션 시작...")
        print(f"  - 이미지 경로: {image_dir}")
        print(f"  - 체스보드 패턴: {chessboard_size}")
        print(f"  - 사각형 크기: {square_size} mm")
        
        # 3D 월드 좌표점 생성 (체스보드 격자점의 실제 좌표)
        # (0,0,0), (1,0,0), (2,0,0) ...., (cols-1, rows-1, 0)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # 코너점 저장 리스트
        objpoints = [] # 3D points in real world space
        imgpoints_l = [] # 2D points in left image plane
        imgpoints_r = [] # 2D points in right image plane
        
        # 이미지 파일 목록 가져오기
        left_images = sorted(glob.glob(os.path.join(image_dir, "left_*.jpg")))
        right_images = sorted(glob.glob(os.path.join(image_dir, "right_*.jpg")))
        
        if len(left_images) != len(right_images):
            print(f"[오류] 왼쪽({len(left_images)})과 오른쪽({len(right_images)}) 이미지 개수가 다릅니다.")
            return False
            
        if len(left_images) < 10:
            print(f"[경고] 이미지가 너무 적습니다 ({len(left_images)}장). 최소 10장 이상 권장합니다.")
            
        valid_pairs = 0
        image_size = None
        
        print("\n[이미지 처리 중...]")
        
        # 커버리지 확인용 이미지 (나중에 초기화)
        coverage_l = None
        coverage_r = None
        
        for left_img_path, right_img_path in zip(left_images, right_images):
            img_l = cv2.imread(left_img_path)
            img_r = cv2.imread(right_img_path)
            
            if img_l is None or img_r is None:
                continue
                
            if image_size is None:
                image_size = (img_l.shape[1], img_l.shape[0])
                # 이미지 크기가 결정되면 커버리지 맵 초기화
                coverage_l = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
                coverage_r = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
                
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            
            # 흔들림(Blur) 체크 - Laplacian Variance
            # 값이 작을수록 흐릿함. 일반적으로 100 미만이면 흐릿하다고 판단하지만
            # 환경에 따라 다르므로 여기서는 로그만 출력하고 낮은 경우 경고
            blur_l = cv2.Laplacian(gray_l, cv2.CV_64F).var()
            blur_r = cv2.Laplacian(gray_r, cv2.CV_64F).var()
            
            # 체스보드 찾기
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, chessboard_size, flags)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, chessboard_size, flags)
            
            is_blurry = blur_l < 50 or blur_r < 50 # 임계값 설정 (조절 가능)
            blur_msg = f" (Blur: L={blur_l:.1f}, R={blur_r:.1f})" if is_blurry else ""
            if is_blurry:
                print(f"  - [Warning] 흔들림 의심: {os.path.basename(left_img_path)}{blur_msg}")
            
            if ret_l and ret_r:
                # 정밀도 향상 (Subpixel)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
                
                # [디버그] 감지된 코너 그리기 및 저장
                debug_dir = os.path.join(image_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                
                vis_l = img_l.copy()
                vis_r = img_r.copy()
                cv2.drawChessboardCorners(vis_l, chessboard_size, corners_l, ret_l)
                cv2.drawChessboardCorners(vis_r, chessboard_size, corners_r, ret_r)
                
                # 커버리지 맵에 그리기 (흰색 점)
                if coverage_l is not None:
                    cv2.drawChessboardCorners(coverage_l, chessboard_size, corners_l, ret_l)
                if coverage_r is not None:
                    cv2.drawChessboardCorners(coverage_r, chessboard_size, corners_r, ret_r)
                
                # 시작점(0번 코너)에 빨간색 큰 원 그리기
                if len(corners_l) > 0:
                    cv2.circle(vis_l, tuple(map(int, corners_l[0][0])), 15, (0, 0, 255), -1)
                if len(corners_r) > 0:
                    cv2.circle(vis_r, tuple(map(int, corners_r[0][0])), 15, (0, 0, 255), -1)
                
                vis_concat = np.hstack([vis_l, vis_r])
                cv2.imwrite(os.path.join(debug_dir, f"vis_{os.path.basename(left_img_path)}"), vis_concat)
                
                objpoints.append(objp)
                imgpoints_l.append(corners_l)
                imgpoints_r.append(corners_r)
                valid_pairs += 1
                print(f"  - OK: {os.path.basename(left_img_path)}{blur_msg}")
            else:
                print(f"  - Skip (감지 실패): {os.path.basename(left_img_path)}")
        
        # 커버리지 이미지 저장
        if coverage_l is not None:
            cv2.imwrite(os.path.join(image_dir, "coverage_left.jpg"), coverage_l)
        if coverage_r is not None:
            cv2.imwrite(os.path.join(image_dir, "coverage_right.jpg"), coverage_r)
        
        if coverage_l is not None:
             print(f"\n[정보] 커버리지 맵 저장됨: coverage_left.jpg, coverage_right.jpg (화면을 골고루 덮었는지 확인하세요)")
                
        print(f"\n총 {len(left_images)}장 중 {valid_pairs}장 유효함.")
        
        if valid_pairs < 5:
            print("[실패] 유효한 이미지가 너무 적어 캘리브레이션을 수행할 수 없습니다.")
            return False
            
        # --- 1. 개별 카메라 캘리브레이션 (Intrinsic) ---
        print("\n[개별 카메라 캘리브레이션 수행 중...]")
        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
            objpoints, imgpoints_l, image_size, None, None
        )
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
            objpoints, imgpoints_r, image_size, None, None
        )
        
        print(f"  - Left Reprojection Error: {ret_l:.4f}")
        print(f"  - Right Reprojection Error: {ret_r:.4f}")
        
        # --- 2. 스테레오 캘리브레이션 (Extrinsic) ---
        print("\n[스테레오 캘리브레이션 수행 중...]")
        # Intrinsic 파라미터를 초기값으로 사용하되, 스테레오 캘리브레이션 과정에서 미세 조정 허용
        # Reprojection Error가 매우 클 때 유용함
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            mtx_l, dist_l,
            mtx_r, dist_r,
            image_size,
            criteria=criteria,
            flags=flags
        )
        
        print(f"  - Stereo Reprojection Error: {ret_stereo:.4f}")
        print(f"  - Translation Vector (T):\n{T}")
        
        if ret_stereo > 1.0:
            print("\n[경고] Reprojection Error가 1.0보다 큽니다. 캘리브레이션 품질이 낮습니다.")
            print("  - 체스보드 촬영 각도를 다양하게 하여 재촬영하거나,")
            print("  - 흔들린 이미지를 제거하고 다시 시도하세요.")
        
        # --- 3. 결과 저장 ---
        # Numpy array는 yaml로 바로 저장이 안되므로 리스트로 변환
        data = {
            'camera_matrix_left': mtx_l.tolist(),
            'dist_coeffs_left': dist_l.tolist(),
            'camera_matrix_right': mtx_r.tolist(),
            'dist_coeffs_right': dist_r.tolist(),
            'R': R.tolist(),
            'T': T.tolist(),
            'E': E.tolist(),
            'F': F.tolist(),
            'image_size': image_size,
            'reprojection_error': ret_stereo
        }
        
        with open(output_file, 'w') as f:
            yaml.dump(data, f)
            
        print(f"\n[성공] 결과가 {output_file}에 저장되었습니다.")
        return True
    
    elif mode == "intrinsic":
        # Intrinsic만 수행 (왼쪽과 오른쪽 각각)
        print("Intrinsic 캘리브레이션 모드")
        left_success = run_intrinsic_calibration(
            image_dir=image_dir,
            output_file="intrinsic_calibration_left.yaml",
            square_size=square_size,
            chessboard_size=chessboard_size,
            camera_side="left"
        )
        right_success = run_intrinsic_calibration(
            image_dir=image_dir,
            output_file="intrinsic_calibration_right.yaml",
            square_size=square_size,
            chessboard_size=chessboard_size,
            camera_side="right"
        )
        return left_success and right_success
    
    elif mode == "extrinsic":
        # Extrinsic만 수행 (Intrinsic 파일 필요)
        return run_extrinsic_calibration(
            image_dir=image_dir,
            intrinsic_left_file="intrinsic_calibration_left.yaml",
            intrinsic_right_file="intrinsic_calibration_right.yaml",
            output_file=output_file,
            square_size=square_size,
            chessboard_size=chessboard_size
        )
    
    else:
        print(f"[오류] 알 수 없는 모드: {mode}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='스테레오 캘리브레이션 도구')
    parser.add_argument('--dir', type=str, default='data/calibration_images', help='이미지 폴더 경로')
    parser.add_argument('--out', type=str, default='calibration_result.yaml', help='결과 파일 경로')
    parser.add_argument('--size', type=float, default=30.0, help='체스보드 사각형 한 변의 길이 (mm)')
    parser.add_argument('--cols', type=int, default=8, help='내부 코너 가로 개수')
    parser.add_argument('--rows', type=int, default=5, help='내부 코너 세로 개수')
    parser.add_argument('--mode', type=str, default='combined', 
                       choices=['combined', 'intrinsic', 'extrinsic'],
                       help='캘리브레이션 모드: combined (통합), intrinsic (Intrinsic만), extrinsic (Extrinsic만)')
    
    args = parser.parse_args()
    
    run_calibration(
        image_dir=args.dir,
        output_file=args.out,
        square_size=args.size,
        chessboard_size=(args.cols, args.rows),
        mode=args.mode
    )
