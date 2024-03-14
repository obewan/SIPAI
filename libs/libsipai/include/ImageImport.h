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
#include <execution>
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
    const int channels = image.channels();
    const int rows = image.rows;
    const int cols = image.cols;
    const int totalPixels = rows * cols;

    std::vector<RGBA> rgbaValues(totalPixels);

    /// lambda helper
    auto convertPixel = [channels](const cv::Vec4b &pixel) {
      return RGBA{{pixel[0] / 255.0f, pixel[1] / 255.0f, pixel[2] / 255.0f,
                   channels == 4 ? pixel[3] / 255.0f : 1.0f}};
    };

    /// std::execution::par_unseq enables parallel execution of the
    /// transformation while relaxing the requirement for sequential execution
    /// order
    auto pixelIterator = image.begin<cv::Vec4b>();
    std::transform(std::execution::par_unseq, pixelIterator,
                   pixelIterator + totalPixels, rgbaValues.begin(),
                   convertPixel);
    return rgbaValues;
  }
};
} // namespace sipai