/**
 * @file ImageHelper.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Image helper
 * @date 2024-03-09
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "RGBA.h"
#include "exception/ImageHelperException.h"
#include <cstddef>
#include <exception>
#include <execution>
#include <opencv2/opencv.hpp>
#include <vector>

namespace sipai {
class ImageHelper {
public:
  /**
   * @brief Reads an image from a specified path and returns it as an OpenCV
   * Mat.
   *
   * @param imagePath The file path of the image to be loaded.
   * @return cv::Mat The imported image as an OpenCV Mat.
   */
  cv::Mat loadImage(const std::string &imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw ImageHelperException("Could not open or find the image: " +
                                 imagePath);
    }
    return image;
  }

  /**
   * @brief Save an OpenCV Matas image.
   *
   * @param imagePath The file path of the image to be saved.
   * @param image The OpenCV Mat to save.
   */
  void saveImage(const std::string &imagePath, cv::Mat &image) {
    try {
      cv::imwrite(imagePath, image);
    } catch (std::exception &ex) {
      throw ImageHelperException(ex.what());
    }
  }

  /**
   * @brief Converts an OpenCV Mat image into a vector of RGBA values.
   *
   * @param image The OpenCV Mat image to be converted.
   * @return std::vector<RGBA> The converted image as a vector of RGBA values.
   */
  std::vector<RGBA> convertToRGBAVector(const cv::Mat &image) {
    const int channels = image.channels();
    const int rows = image.rows;
    const int cols = image.cols;
    const int totalPixels = rows * cols;

    std::vector<RGBA> rgbaValues(totalPixels);

    /// lambda helper
    auto convertPixel = [channels](const cv::Vec4b &pixel) {
      return RGBA{}.fromVec4b(pixel, channels == 4);
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

  /**
   * @brief Converts a vector of RGBA values into an OpenCV Mat image.
   *
   * @param image The vector of RGBA values to be converted.
   * @param size_x The width of the image represented by the vector of RGBA
   * values.
   * @param size_y The height of the image represented by the vector of RGBA
   * values.
   * @return cv::Mat The converted image as an OpenCV Mat.
   */
  cv::Mat convertToMat(const std::vector<RGBA> &image, size_t size_x,
                       size_t size_y) {
    cv::Mat dest(size_y, size_x, CV_8UC4);
    auto destPtr = dest.begin<cv::Vec4b>();
    std::transform(std::execution::par_unseq, image.begin(), image.end(),
                   destPtr, [](const RGBA &rgba) { return rgba.toVec4b(); });
    return dest;
  }
};
} // namespace sipai