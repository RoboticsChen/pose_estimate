project(pose_estimate)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_BUILD_TYPE Debug)

include_directories(pose_estimate /usr/local/include/opencv4)
include_directories(pose_estimate ${CMAKE_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)

add_executable(pose_estimate pose_estimate.cpp realsense.cpp)

target_link_libraries(pose_estimate ${OpenCV_LIBS} realsense2 pthread)
