# Pose Estimate

## Requirements

### Realsense

```shell
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense && mkdir build && cd build && cmake .. && make -j32
sudo make install
cd ..
```

### opencv 

```shell
git clone https://github.com/opencv/opencv
cd opencv && mkdir build && cd build && cmake .. && make -j12
sudo make install
cd ..
```

## Python Requirements

```shell
pip install opencv-python numpy pyrealsense2 glob
```

## Usage
```shell
cd pose_estimate && mkdir build && cd build && cmake .. && make -j4
./pose_estimate
```
> 如需修改分辨率请修改realsense.hpp中的宏定义，支持的分辨率和帧率组合请参考realsense-viewer
