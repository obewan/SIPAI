#include "ImageHelper.h"
#include "Common.h"
#include "exception/ImageHelperException.h"
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

using namespace sipai;

ImageParts ImageHelper::loadImage(const std::string &imagePath, size_t split,
                                  bool withPadding, size_t resize_x,
                                  size_t resize_y) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }

  // Load the image
  cv::Mat mat =
      cv::imread(imagePath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  if (mat.empty()) {
    throw ImageHelperException("Could not open or find the image: " +
                               imagePath);
  }

  // Ensure the image is in BGR format
  if (mat.channels() == 1) {
    cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGRA);
  } else if (mat.channels() == 3) {
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGRA);
  } else if (mat.channels() == 4) {
    cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
  }

  // If the image has only 3 channels (BGR), create and merge an alpha channel
  if (mat.channels() == 3) {
    cv::Mat alphaMat(mat.size(), CV_8UC1, cv::Scalar(255));
    std::vector<cv::Mat> channels{mat, alphaMat};
    cv::Mat bgraMat;
    cv::merge(channels, bgraMat);
    mat = bgraMat;
  }

  // Convert to floating-point range [0, 1] with 4 channels
  mat.convertTo(mat, CV_32FC4, 1.0 / 255.0);
  if (mat.channels() != 4) {
    throw std::runtime_error("incorrect image channels");
  }

  // cv::imshow("Original Image step 3", mat);
  // cv::waitKey(1000 * 60 * 2);

  ImageParts imagesParts;
  auto matParts = splitImage(mat, split, withPadding);
  for (auto &matPart : matParts) {
    if (resize_x > 0 && resize_y > 0) {
      cv::resize(matPart, matPart, cv::Size((int)resize_x, (int)resize_y));
    }
    auto image = std::make_unique<cv::Mat>(matPart);
    imagesParts.push_back(std::move(image));
  }

  // Rq. C++ use Return Value Optimization (RVO) to avoid the extra copy or move
  // operation associated with the return.
  return imagesParts;
}

ImageParts ImageHelper::generateInputImage(const ImageParts &targetImage,
                                           size_t reduce_factor,
                                           size_t resize_x,
                                           size_t resize_y) const {
  ImageParts imagesParts;
  for (auto &targetPart : targetImage) {
    // clone of the Target image to the Input image
    cv::Mat inputPart = targetPart->clone();

    // reduce the resolution of the input image
    cv::Size s = inputPart.size();
    if (reduce_factor != 0) {
      int new_width = (int)(s.width * (1.0f / (float)reduce_factor));
      int new_height = (int)(s.height * (1.0f / (float)reduce_factor));
      if (new_height > 0 && new_height > 0) {
        cv::resize(inputPart, inputPart, cv::Size(new_width, new_height));
      }
    }

    // then resize to the layer resolution
    if (resize_x > 0 && resize_y > 0) {
      cv::resize(inputPart, inputPart, cv::Size((int)resize_x, (int)resize_y));
    }

    // finally convert back to Image
    auto image = std::make_unique<cv::Mat>(inputPart);
    imagesParts.push_back(std::move(image));
  }

  return imagesParts;
}

std::vector<cv::Mat> ImageHelper::splitImage(const cv::Mat &inputImage,
                                             size_t split,
                                             bool withPadding) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }

  std::vector<cv::Mat> outputImages;

  if (split == 1) {
    outputImages.push_back(inputImage);
    return outputImages;
  }

  // Calculate the size of each part in pixels
  int partSizeX = (int)((inputImage.cols + split - 1) / split);
  int partSizeY = (int)((inputImage.rows + split - 1) / split);

  // Calculate the number of splits in x and y directions
  int splitsX = (inputImage.cols + partSizeX - 1) / partSizeX;
  int splitsY = (inputImage.rows + partSizeY - 1) / partSizeY;

  cv::Mat paddedImage;
  if (withPadding) {
    // Calculate the size of padding to make the image size a multiple of
    // partSize
    int paddingX = splitsX * partSizeX - inputImage.cols;
    int paddingY = splitsY * partSizeY - inputImage.rows;

    // Create a copy of the image with padding black (cv::Scalar(0,0,0)) on the
    // right and bottom.
    cv::copyMakeBorder(inputImage, paddedImage, 0, paddingY, 0, paddingX,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  }
  // Loop over the image and create the smaller Region of Interest (roi) parts
  for (int i = 0; i < splitsY; ++i) {
    for (int j = 0; j < splitsX; ++j) {
      int roiWidth =
          (j == (int)split - 1) ? inputImage.cols - j * partSizeX : partSizeX;
      int roiHeight =
          (i == (int)split - 1) ? inputImage.rows - i * partSizeY : partSizeY;
      cv::Rect roi(j * partSizeX, i * partSizeY, roiWidth, roiHeight);
      outputImages.push_back(withPadding ? paddedImage(roi).clone()
                                         : inputImage(roi).clone());
    }
  }

  return outputImages;
}

void ImageHelper::saveImage(const std::string &imagePath,
                            const ImageParts &imageParts, size_t split,
                            size_t resize_x, size_t resize_y) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  try {

    auto mat = joinImages(imageParts, (int)split, (int)split);

    if (mat.empty()) {
      throw ImageHelperException("Image data is empty.");
    }

    if (resize_x > 0 && resize_y > 0) {
      cv::resize(mat, mat, cv::Size((int)resize_x, (int)resize_y));
    }

    // convert back the [0,1] float range image to 255 pixel values
    // TODO: check the original image rtype, CV_8U used temporary
    mat.convertTo(mat, CV_8U, 255.0);

    // TODO: check the original file channel, and use RGBA if it had a proper
    // one, like cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
    cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB); // will reduce to 3 channels

    if (mat.empty()) {
      throw ImageHelperException("Image data is empty.");
    }

    // write the image
    // std::vector<int> params;
    // params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    // params.push_back(9); // Compression level
    // if (!cv::imwrite(imagePath, mat, params)) {
    if (!cv::imwrite(imagePath, mat)) {
      throw ImageHelperException("Error saving image: " + imagePath);
    }
  } catch (ImageHelperException &ihe) {
    throw ihe;
  } catch (std::exception &ex) {
    throw ImageHelperException(ex.what());
  }
}

cv::Mat ImageHelper::joinImages(const ImageParts &images, int splitsX,
                                int splitsY) const {
  if (splitsX == 0 || splitsY == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  std::vector<cv::Mat> rows;
  for (int i = 0; i < splitsY; ++i) {
    std::vector<cv::Mat> row;
    for (int j = 0; j < splitsX; ++j) {
      row.push_back(*images[i * splitsX + j]);
    }
    cv::Mat rowImage;
    cv::hconcat(row, rowImage);
    rows.push_back(rowImage);
  }
  cv::Mat result;
  cv::vconcat(rows, result);
  return result;
}

float ImageHelper::computeLoss(const cv::Mat &outputData,
                               const cv::Mat &targetData) const {
  if (outputData.total() != targetData.total() || outputData.total() == 0 ||
      targetData.total() == 0) {
    throw std::invalid_argument("Output and target images have different "
                                "sizes, or some are empty.");
  }

  // Calculate element-wise squared differences
  cv::Mat diff;
  cv::absdiff(outputData, targetData, diff);
  diff = diff.mul(diff);

  // Compute the sum of squared differences
  cv::Scalar sumSquaredDiff = cv::sum(diff);

  // Compute the number of pixels
  int numPixels = outputData.total();

  // Calculate the MSE loss
  double mseLoss = sumSquaredDiff.val[0] / static_cast<double>(numPixels);

  return mseLoss;
}