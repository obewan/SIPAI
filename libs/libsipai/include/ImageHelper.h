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

#include "Common.h"
#include "Image.h"
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
   * @param resize_x Optional resize the imported image on X (width)
   * @param resize_y Optional resize the imported image on Y (height)
   * @return Image The imported image, optionally resized.
   */
  Image loadImage(const std::string &imagePath, size_t resize_x = 0,
                  size_t resize_y = 0) const;

  /**
   * @brief Save an OpenCV Matas image.
   *
   * @param imagePath The file path of the image to be saved.
   * @param image The Image to save.
   * @param resize_x Optional resize the exported image on X (width)
   * @param resize_y Optional resize the exported image on Y (height)
   */
  void saveImage(const std::string &imagePath, const Image &image,
                 size_t resize_x = 0, size_t resize_y = 0) const;

  /**
   * @brief Converts an OpenCV Mat image into a vector of RGBA values.
   *
   * @param image The OpenCV Mat image to be converted.
   * @return std::vector<RGBA> The converted image as a vector of RGBA values.
   */
  std::vector<RGBA> convertToRGBAVector(const cv::Mat &image) const;

  /**
   * @brief Converts a vector of RGBA values into an OpenCV Mat image.
   *
   * @param image The vector of RGBA values to be converted.
   * @return cv::Mat The converted image as an OpenCV Mat.
   */
  cv::Mat convertToMat(const Image &image) const;

  /**
   * @brief Computes the loss between the output image
   * and the target image. The smaller loss, the better.
   *
   * @param outputData The output image data produced by the neural network.
   * @param targetData The expected target image data.
   *
   * @return The computed loss.
   */
  float computeLoss(const std::vector<RGBA> &outputData,
                    const std::vector<RGBA> &targetData) const;
};
} // namespace sipai