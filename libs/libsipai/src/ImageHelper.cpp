#include "ImageHelper.h"
#include "exception/ImageHelperException.h"
#include <opencv2/imgcodecs.hpp>

using namespace sipai;

ImageParts ImageHelper::loadImage(const std::string &imagePath, size_t split,
                                  bool withPadding, size_t resize_x,
                                  size_t resize_y) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  cv::Mat mat = cv::imread(imagePath, cv::IMREAD_COLOR);
  if (mat.empty()) {
    throw ImageHelperException("Could not open or find the image: " +
                               imagePath);
  }

  ImageParts imagesParts;
  auto matParts = splitImage(mat, split, withPadding);
  for (auto &matPart : matParts) {
    cv::Size s = matPart.size();

    if (resize_x > 0 && resize_y > 0) {
      cv::resize(matPart, matPart, cv::Size(resize_x, resize_y));
    } else {
      resize_x = s.width;
      resize_y = s.height;
    }

    const auto &data = convertToRGBAVector(matPart);
    imagesParts.emplace_back(data, resize_x, resize_y, s.width, s.height);
  }

  // Rq. C++ use Return Value Optimization (RVO) to avoid the extra copy or move
  // operation associated with the return.
  return imagesParts;
}

std::vector<cv::Mat> ImageHelper::splitImage(const cv::Mat &inputImage,
                                             size_t split,
                                             bool withPadding) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  std::vector<cv::Mat> outputImages;

  // Calculate the size of each part in pixels
  int partSizeX = (inputImage.cols + split - 1) / split;
  int partSizeY = (inputImage.rows + split - 1) / split;

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
    std::vector<cv::Mat> mats(imageParts.size());
    std::transform(imageParts.begin(), imageParts.end(), mats.begin(),
                   [&](const Image &part) { return convertToMat(part); });
    auto mat = joinImages(mats, split, split);

    if (resize_x > 0 && resize_y > 0) {
      cv::resize(mat, mat, cv::Size(resize_x, resize_y));
    }
    cv::imwrite(imagePath, mat);
  } catch (std::exception &ex) {
    throw ImageHelperException(ex.what());
  }
}

cv::Mat ImageHelper::joinImages(const std::vector<cv::Mat> &images, int splitsX,
                                int splitsY) const {
  if (splitsX == 0 || splitsY == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  std::vector<cv::Mat> rows;
  for (int i = 0; i < splitsY; ++i) {
    std::vector<cv::Mat> row;
    for (int j = 0; j < splitsX; ++j) {
      row.push_back(images[i * splitsX + j]);
    }
    cv::Mat rowImage;
    cv::hconcat(row, rowImage);
    rows.push_back(rowImage);
  }
  cv::Mat result;
  cv::vconcat(rows, result);
  return result;
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
  if (outputData.size() != targetData.size() || outputData.size() == 0 ||
      targetData.size() == 0) {
    throw std::invalid_argument("Output and target images must have the same "
                                "size, or size is null.");
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