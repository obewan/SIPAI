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
#include "Data.h"
#include "exception/ImageHelperException.h"
#include <cstddef>
#include <exception>
#include <execution>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace sipai {
class ImageHelper {
public:
  /**
   * @brief Reads an image from a specified path and returns it as Image
   * parts
   *
   * @param imagePath The file path of the image to be loaded.
   * @param split The split factor.
   * @param withPadding Add padding to the splitted image parts.
   * @param resize_x Optional resize the imported image on X (width).
   * @param resize_y Optional resize the imported image on Y (height).
   * @return Image The imported image parts, optionally resized.
   */
  ImageParts loadImage(const std::string &imagePath, size_t split,
                       bool withPadding, size_t resize_x = 0,
                       size_t resize_y = 0) const;

  /**
   * @brief Generate an input image from a target image
   *
   * @param targetImage
   * @param reduce_factor
   * @param resize_x
   * @param resize_y
   * @return ImageParts
   */
  ImageParts generateInputImage(const ImageParts &targetImage,
                                size_t reduce_factor, size_t resize_x = 0,
                                size_t resize_y = 0) const;

  /**
   * @brief Split an OpenCV Mat to smaller Mats.
   *
   * @param inputImage The OpenCV Mat to split.
   * @param split The split factor.
   * @param withPadding Add padding if split is not a multiple of image width
   * and height
   * @return std::vector<cv::Mat>
   */
  std::vector<cv::Mat> splitImage(const cv::Mat &inputImage, size_t split,
                                  bool withPadding = false) const;

  /**
   * @brief Save an image.
   *
   * @param imagePath The file path of the image to be saved.
   * @param imageParts The Image to save.
   * @param split The split factor.
   * @param resize_x Optional resize the exported image on X (width)
   * @param resize_y Optional resize the exported image on Y (height)
   */
  void saveImage(const std::string &imagePath, const ImageParts &imageParts,
                 size_t split, size_t resize_x = 0, size_t resize_y = 0) const;

  /**
   * @brief Concat the images parts into one image.
   *
   * @param images OpenCV image parts
   * @param splitsX
   * @param splitsY
   * @return Image
   */
  Image joinImages(const ImageParts &images, int splitsX, int splitsY) const;

  /**
   * @brief Computes the loss between the output image
   * and the target image. The smaller loss, the better.
   *
   * @param outputData The output image data produced by the neural network.
   * @param targetData The expected target image data.
   *
   * @return The computed loss.
   */
  float computeLoss(const cv::Mat &outputData, const cv::Mat &targetData) const;

private:
  cv::Mat convertToFloatMat(const cv::Mat &mat) const;
};
} // namespace sipai