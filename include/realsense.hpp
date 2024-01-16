#include <iostream>
#include <iomanip>
#include <string>
#include <csignal>
#include <thread>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                     These parameters are reconfigurable                                 //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define rgbFORMAT RS2_FORMAT_RGB8  // rs2_format identifies how binary data is encoded within a frame      //
#define depthFORMAT RS2_FORMAT_Z16 // rs2_format identifies how binary data is encoded within a frame      //
#define WIDTH 640                  // Defines the number of columns for each frame                         //
#define HEIGHT 480                 // Defines the number of lines for each frame                           //
#define FPS 30                     // Defines the rate of frames per second                                //
#define STREAM_INDEX 0             // Defines the stream index, used for multiple streams of the same type //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern bool g_signalCtrlC;

typedef struct tagImageData
{
    uint8_t *data; // 图像数据指针
    int width;
    int height;
    double time_stamp;
    std::string image_name;
}ImageData;

class RealsenseCamera
{
private:
    rs2::colorizer colored_depth;
    rs2::config pipe_config;
    rs2::pipeline pipe;

    rs2::frameset frame_data;
    uint8_t *rgb_data;
    double rgb_timestamp;
    std::thread update_thread;

    mutable std::mutex frame_lock;

public:
    RealsenseCamera();
    ~RealsenseCamera();
    void update();
    ImageData get_rgb_frame();
    ImageData get_depth_frame();
};

std::string stamp2string(double timestamp);
void HandleSignal(int sig);