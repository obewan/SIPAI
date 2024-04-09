#include "ImageHelper.h"
#include <opencv2/imgcodecs.hpp>
#include <pstl/glue_execution_defs.h>

using namespace sipai;

Image ImageHelper::loadImage(const std::string &imagePath, size_t resize_x,
                             size_t resize_y) const {
  cv::Mat mat = cv::imread(imagePath, cv::IMREAD_COLOR);
  if (mat.empty()) {
    throw ImageHelperException("Could not open or find the image: " +
                               imagePath);
  }
  cv::Size s = mat.size();

  if (resize_x > 0 && resize_y > 0) {
    cv::resize(mat, mat, cv::Size(resize_x, resize_y));
  } else {
    resize_x = s.width;
    resize_y = s.height;
  }

  const auto &data = convertToRGBAVector(mat);

  // Rq. C++ use Return Value Optimization (RVO) to avoid the extra copy or move
  // operation associated with the return.
  return Image(data, resize_x, resize_y, s.width, s.height);
}

void ImageHelper::saveImage(const std::string &imagePath, const Image &image,
                            size_t resize_x, size_t resize_y) const {
  try {
    auto mat = convertToMat(image);
    if (resize_x > 0 && resize_y > 0) {
      cv::resize(mat, mat, cv::Size(resize_x, resize_y));
    }
    cv::imwrite(imagePath, mat);
  } catch (std::exception &ex) {
    throw ImageHelperException(ex.what());
  }
}

std::vector<RGBA> ImageHelper::convertToRGBAVector(const cv::Mat &mat) const {
  const int channels = mat.channels();
  const int rows = mat.rows;
  const int cols = mat.cols;
  const int totalPixels = rows * cols;

  std::vector<RGBA> rgbaValues(totalPixels);
  auto pixelIterator = mat.begin<cv::Vec4b>();

  std::transform(pixelIterator, pixelIterator + totalPixels, rgbaValues.begin(),
                 [&channels](const cv::Vec4b &pixel) {
                   return RGBA(pixel, channels == 4);
                 });

  return rgbaValues;
}

cv::Mat ImageHelper::convertToMat(const Image &image) const {
  cv::Mat dest(image.size_y, image.size_x, CV_8UC4);
  auto destPtr = dest.begin<cv::Vec4b>();

  std::transform(image.data.begin(), image.data.end(), destPtr,
                 [](const RGBA &rgba) { return rgba.toVec4b(); });

  return dest;
}

float ImageHelper::computeLoss(const std::vector<RGBA> &outputData,
                               const std::vector<RGBA> &targetData) const {
  if (outputData.size() != targetData.size()) {
    throw std::invalid_argument(
        "Output and target images must have the same size.");
  }
  // Using the mean squared error (MSE) loss algorithm
  const auto squaredDifferences = [](const RGBA &a, const RGBA &b) {
    return (a - b).pow(2).sum();
  };

  const float totalLoss = std::inner_product(
      outputData.begin(), outputData.end(), targetData.begin(), 0.0f,
      std::plus<>(), squaredDifferences);

  return totalLoss / (outputData.size() * 4);
}