#include "realsense.hpp"
bool g_signalCtrlC = false;

// int main() {
//   RealsenseCamera camera;
//   // std::thread update_thread([&camera]()
//   //                           { camera.update(); });
//   // std::this_thread::sleep_for(std::chrono::milliseconds(10000));
//   while (!g_signalCtrlC) {
//     auto rgb = camera.get_rgb_frame();
//     cv::Mat color_image(rgb.height, rgb.width, CV_8UC3, rgb.data);
//     cv::Mat bgr_image(0, 0, CV_8UC3, 0);
//     cv::cvtColor(color_image, bgr_image, cv::COLOR_RGB2BGR);
//     cv::imshow("color_Image", bgr_image);

//     auto depth = camera.get_depth_frame();
//     cv::Mat depth_image(depth.height, depth.width, CV_8UC1, depth.data);
//     cv::equalizeHist(depth_image, depth_image);
//     cv::Mat colored_depth(0, 0, CV_8UC1, 0);
//     cv::applyColorMap(depth_image, colored_depth, cv::COLORMAP_JET);
//     cv::imshow("depth_Image", colored_depth);
//     int key = cv::waitKey(10);
//     if (key == 27) {
//       cv::destroyAllWindows();
//       g_signalCtrlC = true;
//       break;
//     }
//     // cv::imwrite(rgb.image_name, color_image);
//   }
//   // update_thread.join();
//   return 0;
// }

void RealsenseCamera::update() {
  while (!g_signalCtrlC) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    // std::cout << "." << std::flush;
    std::lock_guard<std::mutex> lock(frame_lock);
    frame_data = pipe.wait_for_frames();
  }
}

ImageData RealsenseCamera::get_rgb_frame() {
  ImageData image;
  std::lock_guard<std::mutex> lock(frame_lock);
  // frame_data = pipe.wait_for_frames();
  rs2::frame color = frame_data.get_color_frame();
  double color_time_stamp = color.get_timestamp();
  // std::cout << stamp2string(color_time_stamp) << std::endl;
  image.data =
      (uint8_t *)const_cast<void *>(color.as<rs2::video_frame>().get_data());
  image.width = color.as<rs2::video_frame>().get_width();
  image.height = color.as<rs2::video_frame>().get_height();
  image.time_stamp = color_time_stamp;
  image.image_name = stamp2string(color_time_stamp) + ".jpg";
  return image;
}

ImageData RealsenseCamera::get_depth_frame() {
  ImageData image;
  std::lock_guard<std::mutex> lock(frame_lock);
  // frame_data = pipe.wait_for_frames();
  rs2::frame depth = frame_data.get_depth_frame();
  double depth_time_stamp = depth.get_timestamp();
  // std::cout << stamp2string(depth_time_stamp) << std::endl;
  image.data =
      (uint8_t *)const_cast<void *>(depth.as<rs2::depth_frame>().get_data());
  image.width = depth.as<rs2::depth_frame>().get_width();
  image.height = depth.as<rs2::depth_frame>().get_height();
  image.time_stamp = depth_time_stamp;
  image.image_name = stamp2string(depth_time_stamp) + ".jpg";
  return image;
}

RealsenseCamera::RealsenseCamera() {
  pipe_config.enable_stream(RS2_STREAM_DEPTH, WIDTH, HEIGHT, depthFORMAT, FPS);
  pipe_config.enable_stream(RS2_STREAM_COLOR, WIDTH, HEIGHT, rgbFORMAT, FPS);
  rs2::pipeline_profile profile = pipe.start(pipe_config);
  frame_data = pipe.wait_for_frames();
  update_thread = std::thread([this]() { update(); });
}

RealsenseCamera::~RealsenseCamera() {
  g_signalCtrlC = true;
  update_thread.join();
}

std::string stamp2string(double timestamp) {
  // 将时间戳转化为秒和毫秒
  time_t seconds = static_cast<time_t>(timestamp / 1000);
  long milliseconds = static_cast<long>((timestamp) - (seconds * 1000));

  // 将时间戳转化为tm结构体
  struct tm *timeinfo;
  timeinfo = localtime(&seconds);

  // 使用stringstream来格式化输出
  std::stringstream ss;
  ss << (timeinfo->tm_year + 1900) << std::setw(2) << std::setfill('0')
     << (timeinfo->tm_mon + 1) << std::setw(2) << std::setfill('0')
     << timeinfo->tm_mday << '_' << std::setw(2) << std::setfill('0')
     << timeinfo->tm_hour << std::setw(2) << std::setfill('0')
     << timeinfo->tm_min << std::setw(2) << std::setfill('0')
     << timeinfo->tm_sec << '_' << std::setw(3) << std::setfill('0')
     << milliseconds; // 添加毫秒部分

  return ss.str();
}

void HandleSignal(int sig) {
  printf("recieve signal:%d.\n", sig);
  switch (sig) {
  case SIGINT:
    g_signalCtrlC = true;
    printf("  ctrl+c signal ...\n");
    break;
  }
}