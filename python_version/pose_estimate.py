import pyrealsense2 as rs
import numpy as np
import cv2


class camera_config():
    def __init__(self, rgbW, rgbH, rgbF, depthW=None, depthH=None, depthF=None) -> None:
        self.rgbW = rgbW
        self.rgbH = rgbH
        self.rgbF = rgbF
        if depthW is not None and depthH is not None and depthF is not None:
            self.depthW = depthW
            self.depthH = depthH
            self.depthF = depthF
        else:
            self.depthW=640
            self.depthH=480
            self.depthF=15


class RealsenseCamera():

    def __init__(self, config) -> None:
        self.inited: bool = False
        # Image params: list[width,height,fps]
        self.color_config: list[int, int, int] = [
            config.rgbW, config.rgbH, config.rgbH]
        self.depth_config: list[int, int, int] = [
            config.depthW, config.depthH, config.depthH]
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self._INTRINSIC = None
        self.inited = self.init()

    @property
    def WIDTH(self) -> int:
        return self.depth_config[0]

    @property
    def HEIGHT(self) -> int:
        return self.depth_config[1]

    @property
    def INTRINSIC(self) -> np.ndarray:
        return self._INTRINSIC

    def init(self, *args, **kwargs) -> bool:
        config = rs.config()
        config.enable_stream(rs.stream.depth,
                             self.depth_config[0],
                             self.depth_config[1],
                             rs.format.z16,
                             self.depth_config[2])
        config.enable_stream(rs.stream.color,
                             self.color_config[0],
                             self.color_config[1],
                             rs.format.bgr8,
                             self.color_config[2])
        # Start streaming
        cfg = self.pipeline.start(config)
        i = 0
        while i < 50:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            i += 1
        # Fetch stream profile for color stream
        profile = cfg.get_stream(rs.stream.depth)
        print('profile', profile.as_video_stream_profile())
        # Downcast to video_stream_profile and fetch intrinsics
        intr = profile.as_video_stream_profile().get_intrinsics()

        # width, height, ppx, ppy, fx, fy, Brown_Conrady
        self._INTRINSIC = np.array([[intr.fx, 0, intr.ppx],
                                    [0, intr.fy, intr.ppy],
                                    [0, 0, 1]], dtype=np.dtypes.Float64DType)
        return True

    def deinit(self) -> bool:
        self.inited = False
        return True

    def get_rgb(self) -> np.ndarray:
        return self.get_rgb_depth('rgb')

    def get_depth(self) -> np.ndarray:
        return self.get_rgb_depth('depth')

    def get_rgb_depth(self, img_type: str = 'both'):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        if img_type == 'rgb':
            return color_image
        elif img_type == 'depth':
            return depth_image
        else:
            return frames


def pose_estimation(img, mtx, dist, pattern, square_size, rvec_prior, tvec_prior):

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

    # # Crop the image to the region of interest (ROI)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

    # 选择标定板的行列数
    rows, cols = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    # 在图像中查找角点
    ret, corners = cv2.findCirclesGrid(gray, (cols, rows), None)
    if ret:
        cv2.drawChessboardCorners(undistorted_img, (cols, rows), corners, ret)
    rvecs, tvecs = None, None
    if ret:
        # 解项目架
        if rvec_prior is not None and tvec_prior is not None:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                objp, corners, mtx, dist, rvec_prior, tvec_prior, True, flags=cv2.SOLVEPNP_EPNP)  # flags=cv2.SOLVEPNP_UPNP)
        else:
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
                objp, corners, mtx, dist, flags=cv2.SOLVEPNP_EPNP)  # flags=cv2.SOLVEPNP_UPNP)
        # visualize_inliers(undistorted_img, inliers, corners)
        print(len(inliers), end="")
        image = cv2.drawFrameAxes(
            undistorted_img, mtx, dist, rvecs, tvecs, 2*square_size)
        return ret, rvecs, tvecs, image
    else:
        cv2.imwrite("error.jpg", undistorted_img)
        return ret, None, None, undistorted_img





if __name__ == "__main__":
    pattern = (9, 11)
    square_size = 0.03
    config = camera_config(640, 480, 30)
    camera = RealsenseCamera(config)
    mtx = np.array([[600.22426814,   0., 322.13195727],
                    [0., 600.03649697, 243.92724777],
                    [0.,  0.,  1.]])
    dist = np.array(
        [[0.13317231, -0.06212984, 0.00074379, 0.00107854, -0.69059097]])
    rvec_prior, tvec_prior = None, None
    while True:
        color = camera.get_rgb()
        ret, rvec, tvec, image = pose_estimation(
            color, mtx, dist, pattern, square_size, rvec_prior, tvec_prior)
        rvec_prior, tvec_prior = rvec, tvec

        pose_matrix = None
        if ret:
            print(rvec.squeeze().reshape(1, -1), tvec.squeeze().reshape(1, -1))
            # 将旋转矩阵和平移向量组合成位姿变换矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            pose_matrix = np.eye(4, 4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = tvec.T
        # 显示带有坐标轴的图像
        cv2.imshow('Axes', image)
        key = cv2.waitKey(10)
        if key == ord('s'):
            file_path = 'pose_matrix.txt'
            with open(file_path, 'a') as file:
                np.savetxt(file, pose_matrix.reshape(
                    1, -1).squeeze(), fmt='%.3f', delimiter=', ')
        if key == 27:
            cv2.destroyAllWindows()
            break
