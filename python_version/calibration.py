import glob
import os
import cv2
import numpy as np


def calibration(pattern, square_size, image_path):
    # 准备标定板的参数
    rows, cols = pattern

    # 生成标定板上点的坐标
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    # 存储物体点和图像点的列表
    objpoints = []  # 3D点（物体点）的集合
    imgpoints = []  # 2D点（图像点）的集合

    # 读取图像并检测角点
    images = glob.glob(os.path.join(image_path, '*.png'))
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findCirclesGrid(gray, (cols, rows), None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, (cols, rows), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(10)

    cv2.destroyAllWindows()

    print(gray.shape[::-1])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

if __name__ == "__main__":
    pattern  = (9,11)
    square_size = 0.03
    camera_type = "Realsense"
    image_quality = "480p"
    image_path = f"images/{image_quality}"
    mtx, dist = calibration(pattern,square_size,image_path)
    # 保存标定结果
    np.savez(f'{camera_type}-{image_quality}.npz', mtx=mtx, dist=dist)

    # 读取标定结果
    # data = np.load('calibration.npz')
    # mtx, dist, rvecs, tvecs = data['mtx'], data['dist'], data['rvecs'], data['tvecs']
