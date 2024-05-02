/**
 * @file Image.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Image data
 * @date 2024-04-30
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#include <cstddef>
#include <memory>
#include <opencv2/opencv.hpp>

namespace sipai {
struct Image {
  cv::Mat data;
  size_t height; // original height
  size_t width;  // original width
  int type;      // original type
  int channels;  // original channels

  void resize(size_t width, size_t height) {
    if (width > 0 && height > 0) {
      cv::resize(data, data, cv::Size((int)width, (int)height));
    }
  }
};
} // namespace sipai