/**
 * @file ImageImport.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Image import
 * @date 2024-03-09
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "RGBA.h"
#include "exception/ImageImportException.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace sipai {
class ImageImport {
public:
  cv::Mat importImage(const std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw ImageImportException("Could not open or find the image: " +
                                 imagePath);
    }
    return image;
  }

  std::vector<RGBA> convertToRGBAVector(const cv::Mat &image) {
    std::vector<RGBA> rgbaValues;
    image.forEach<cv::Vec4b>(
        [&rgbaValues](const cv::Vec4b &pixel, const int *position) {
          rgbaValues.push_back({pixel[0] / 255.0f, pixel[1] / 255.0f,
                                pixel[2] / 255.0f, pixel[3] / 255.0f});
        });
    return rgbaValues;
  }
};
} // namespace sipai