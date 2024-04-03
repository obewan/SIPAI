#include "ImageHelper.h"
#include "Manager.h"
#include <pstl/glue_execution_defs.h>

using namespace sipai;

cv::Mat ImageHelper::loadImage(const std::string &imagePath) {
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw ImageHelperException("Could not open or find the image: " +
                               imagePath);
  }
  return image;
}

void ImageHelper::saveImage(const std::string &imagePath, cv::Mat &image) {
  try {
    cv::imwrite(imagePath, image);
  } catch (std::exception &ex) {
    throw ImageHelperException(ex.what());
  }
}

std::vector<RGBA> ImageHelper::convertToRGBAVector(const cv::Mat &image) {
  const int channels = image.channels();
  const int rows = image.rows;
  const int cols = image.cols;
  const int totalPixels = rows * cols;

  std::vector<RGBA> rgbaValues(totalPixels);
  auto pixelIterator = image.begin<cv::Vec4b>();

  /// std::execution::par_unseq enables parallel execution of the
  /// transformation while relaxing the requirement for sequential execution
  /// order
  auto processPixels = [&rgbaValues, &channels](auto executionPolicy,
                                                auto begin, auto end) {
    std::transform(executionPolicy, begin, end, rgbaValues.begin(),
                   [&channels](const cv::Vec4b &pixel) {
                     return RGBA(pixel, channels == 4);
                   });
  };

  // switch between sequential or parallel processing,
  // beware there will be lots of threads
  if (Manager::getInstance().app_params.enable_parallax) {
    processPixels(std::execution::par_unseq, pixelIterator,
                  pixelIterator + totalPixels);
  } else {
    processPixels(std::execution::seq, pixelIterator,
                  pixelIterator + totalPixels);
  }
  return rgbaValues;
}

cv::Mat ImageHelper::convertToMat(const std::vector<RGBA> &image, size_t size_x,
                                  size_t size_y) {
  cv::Mat dest(size_y, size_x, CV_8UC4);
  auto destPtr = dest.begin<cv::Vec4b>();

  auto processPixels = [&destPtr](auto executionPolicy, auto image) {
    std::transform(std::execution::par_unseq, image.begin(), image.end(),
                   destPtr, [](const RGBA &rgba) { return rgba.toVec4b(); });
  };

  // switch between sequential or parallel processing,
  // beware there will be lots of threads
  if (Manager::getInstance().app_params.enable_parallax) {
    processPixels(std::execution::par_unseq, image);
  } else {
    processPixels(std::execution::seq, image);
  }
  return dest;
}