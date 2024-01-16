import pyrealsense2 as rs
import numpy as np
import cv2


class RealsenseCamera():

    def __init__(self, backend) -> None:
        self.backend = backend
        self.inited: bool = False
        # Image params: list[width,height,fps]
        self.color_config: list[int, int, int] = [640, 480, 30]
        self.depth_config: list[int, int, int] = [640, 480, 30]
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


if __name__ == "__main__":
    backend = ''
    camera = RealsenseCamera(backend)
    print(camera.INTRINSIC)
    align = rs.align(rs.stream.color)
    for i in range(20):
        image = None
        while True:
            frame = camera.get_rgb_depth()
            aligned_frames = align.process(frame)
            aligned_color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(aligned_color_frame.get_data())
            cv2.imshow(f"capture{i}", color_image)
            key = cv2.waitKey(1)
            if key == 27:
                image = color_image
                cv2.destroyAllWindows()
                break
        cv2.imwrite(f"./image/color{i}.png", image)


    # backend = ''
    # camera = RealsenseCamera(backend)
    # rgb = camera.get_rgb()
    # depth = camera.get_depth()
    # print(camera.INTRINSIC)
    # cv2.imwrite("color.png", rgb)
    # cv2.imwrite("depth.png", depth)
    # align = rs.align(rs.stream.color)
    # frame = camera.get_rgb_depth()
    # aligned_frames = align.process(frame)
    # aligned_depth_frame = aligned_frames.get_depth_frame()
    # aligned_color_frame = aligned_frames.get_color_frame()

    # depth_image = np.asanyarray(aligned_depth_frame.get_data())
    # color_image = np.asanyarray(aligned_color_frame.get_data())

    # cv2.imwrite("color_.png", color_image)
    # cv2.imwrite("depth_.png", depth_image)

    # depth_frame = frame.get_depth_frame()
    # hole_filling = rs.hole_filling_filter()
    # filled_depth = hole_filling.process(depth_frame)
    # colorizer = rs.colorizer()
    # colorized_depth = np.asanyarray(
    #     colorizer.colorize(filled_depth).get_data())
    # # plt.imshow(colorized_depth)
    # plt.show()

    # print(camera.INTRINSIC)
    # cv2.imwrite("color9.png", rgb)
    # cv2.imwrite("depth9.png", depth)
