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
  cv::Mat loadImage(const std::string &imagePath);

  /**
   * @brief Save an OpenCV Matas image.
   *
   * @param imagePath The file path of the image to be saved.
   * @param image The OpenCV Mat to save.
   */
  void saveImage(const std::string &imagePath, cv::Mat &image);

  /**
   * @brief Converts an OpenCV Mat image into a vector of RGBA values.
   *
   * @param image The OpenCV Mat image to be converted.
   * @return std::vector<RGBA> The converted image as a vector of RGBA values.
   */
  std::vector<RGBA> convertToRGBAVector(const cv::Mat &image);

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
                       size_t size_y);

  /**
   * @brief Computes the loss between the output image
   * and the target image. The smaller loss, the better.
   *
   * @param outputImage The output image produced by the neural network.
   * @param targetImage The expected target image.
   *
   * @return The computed loss.
   */
  float computeLoss(const std::vector<RGBA> &outputImage,
                    const std::vector<RGBA> &targetImage);
};
} // namespace sipai