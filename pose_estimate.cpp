#include "realsense.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

cv::Mat poseEstimation(const cv::Mat img, const cv::Mat &mtx,
                       const cv::Mat &dist, const cv::Size &pattern,
                       float square_size, cv::Mat &rvec_prior,
                       cv::Mat &tvec_prior) {

  cv::Size img_size = img.size();
  cv::Mat new_camera_matrix, undistorted_img;
  cv::Rect validPixROI;

  new_camera_matrix = cv::getOptimalNewCameraMatrix(mtx, dist, img_size, 1,
                                                    img_size, &validPixROI);
  cv::undistort(img, undistorted_img, mtx, dist, new_camera_matrix);
  cv::Mat cropped_image = undistorted_img(validPixROI);

  cv::Mat gray;

  cv::cvtColor(cropped_image, gray, cv::COLOR_BGR2GRAY);

  int rows = pattern.height;
  int cols = pattern.width;

  std::vector<cv::Point3f> objp;

  objp.reserve(rows * cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      objp.push_back(cv::Point3f(j * square_size, i * square_size, 0.0f));
    }
  }

  std::vector<cv::Point2f> corners;

  bool found = cv::findCirclesGrid(gray, pattern, corners);
  if (found) {
    cv::drawChessboardCorners(cropped_image, pattern, corners, found);
  }

  cv::Mat rvec, tvec, inliers;
  if (found) {

    if (!rvec_prior.empty() && !tvec_prior.empty()) {
      cv::solvePnPRansac(objp, corners, mtx, dist, rvec_prior, tvec_prior, true,
                         100, 8.0F, 0.99, inliers, cv::SOLVEPNP_EPNP);
    } else {
      cv::solvePnPRansac(objp, corners, mtx, dist, rvec, tvec, false, 100, 8.0F,
                         0.99, inliers, cv::SOLVEPNP_EPNP);
      rvec.copyTo(rvec_prior);
      tvec.copyTo(tvec_prior);
    }

    cv::drawFrameAxes(cropped_image, mtx, dist, rvec_prior, tvec_prior,
                      2 * square_size);

    std::cout << inliers.rows << " ";

  } else {
    rvec.copyTo(rvec_prior);
    tvec.copyTo(tvec_prior);
    cv::imwrite("../logs/error.png", cropped_image);
  }

  return cropped_image;
}

int main() {
  cv::Size pattern(11, 9);
  float square_size = 0.03;
  RealsenseCamera camera;
  //注：这是realsense 640x480 分辨率下的内参矩阵和畸变参数，
  //如果用别的分辨率请用pytho_version中的标定程序重新标定
  cv::Mat mtx = (cv::Mat_<double>(3, 3) 
                << 600.22426814, 0., 322.13195727, 
                    0., 600.03649697, 243.92724777,
                     0., 0., 1.);
  cv::Mat dist = (cv::Mat_<double>(1, 5) 
                << 0.13317231, -0.06212984, 0.00074379,
                  0.00107854, -0.69059097);

  cv::Mat rvec_prior, tvec_prior;

  while (true) {
    auto rgb = camera.get_rgb_frame();
    cv::Mat color(rgb.height, rgb.width, CV_8UC3, rgb.data);
    cv::Mat bgr_image(0, 0, CV_8UC3, 0);
    cv::cvtColor(color, bgr_image, cv::COLOR_RGB2BGR);

    cv::Mat image = poseEstimation(bgr_image, mtx, dist, pattern, square_size,
                                   rvec_prior, tvec_prior);

    if (rvec_prior.cols) {

      // std::cout << "rvec:" << rvec_prior.reshape(3, 1)
      //           << "; tvec:" << tvec_prior.reshape(3, 1) << std::endl;
      // 设置输出格式为固定小数点，保留五位小数
      std::cout << std::fixed << std::setprecision(5);

      // 输出rvec
      std::cout << "rvec:[";
      for (int i = 0; i < rvec_prior.rows; ++i) {
        std::cout << rvec_prior.at<double>(i, 0) << ", ";
      }
      std::cout << "]; ";

      // 输出tvec
      std::cout << "tvec:[";
      for (int i = 0; i < tvec_prior.rows; ++i) {
        std::cout << tvec_prior.at<double>(i, 0) << ", ";
      }
      std::cout << "]" << std::endl;

      // Convert rvec to rotation matrix
      cv::Mat rotation_matrix;
      cv::Rodrigues(rvec_prior, rotation_matrix);

      // Create a 4x4 pose matrix
      cv::Mat pose_matrix = cv::Mat::eye(4, 4, CV_64F);

      // Copy rotation matrix to the upper-left 3x3 of pose_matrix
      rotation_matrix.copyTo(pose_matrix(cv::Rect(0, 0, 3, 3)));

      // Copy tvec to the first three elements of the last column of pose_matrix
      tvec_prior.copyTo(pose_matrix(cv::Rect(3, 0, 1, 3)));
    }
    // Display image with axes
    cv::imshow("color", image);

    // Wait for a key event
    int key = cv::waitKey(10);

    // // Save pose_matrix to a file when 's' key is pressed
    // if (key == 's') {
    //   std::ofstream file("pose_matrix.txt", std::ios_base::app);
    //   file << cv::format(pose_matrix.reshape(1, -1), cv::Formatter::FMT_CSV)
    //        << std::endl;
    // }

    // Exit when 'Esc' key is pressed
    if (key == 27) {
      cv::destroyAllWindows();
      break;
    }
  }
  return 0;
}