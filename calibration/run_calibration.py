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

def run_calibration(
    image_dir: str = "data/calibration_images",
    output_file: str = "calibration_result.yaml",
    square_size: float = 30.0,
    chessboard_size: tuple = (8, 5) # (가로 내부 코너 수, 세로 내부 코너 수)
):
    """
    스테레오 캘리브레이션 실행
    
    Args:
        image_dir: 이미지가 저장된 디렉토리
        output_file: 결과 저장 파일 경로
        square_size: 체스보드 사각형 한 변의 길이 (mm)
        chessboard_size: (cols, rows) 내부 코너 개수
    """
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
    for left_img_path, right_img_path in zip(left_images, right_images):
        img_l = cv2.imread(left_img_path)
        img_r = cv2.imread(right_img_path)
        
        if img_l is None or img_r is None:
            continue
            
        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])
            
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
    flags = cv2.CALIB_FIX_INTRINSIC
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='스테레오 캘리브레이션 도구')
    parser.add_argument('--dir', type=str, default='data/calibration_images', help='이미지 폴더 경로')
    parser.add_argument('--out', type=str, default='calibration_result.yaml', help='결과 파일 경로')
    parser.add_argument('--size', type=float, default=24.0, help='체스보드 사각형 한 변의 길이 (mm)')
    parser.add_argument('--cols', type=int, default=8, help='내부 코너 가로 개수')
    parser.add_argument('--rows', type=int, default=5, help='내부 코너 세로 개수')
    
    args = parser.parse_args()
    
    run_calibration(
        image_dir=args.dir,
        output_file=args.out,
        square_size=args.size,
        chessboard_size=(args.cols, args.rows)
    )

