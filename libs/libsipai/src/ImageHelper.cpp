#include "ImageHelper.h"
#include "Common.h"
#include "Data.h"
#include "SimpleLogger.h"
#include "exception/ImageHelperException.h"
#include <filesystem>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <string>

using namespace sipai;

ImageParts ImageHelper::loadImage(const std::string &imagePath, size_t split,
                                  bool withPadding, size_t resize_x,
                                  size_t resize_y) const {
  if (split == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  // Check the path
  if (!std::filesystem::exists(imagePath)) {
    throw ImageHelperException("Could not find the image: " + imagePath);
  }

  // Load the image
  try {
    cv::Mat mat =
        cv::imread(imagePath, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);

    if (mat.empty()) {
      throw ImageHelperException("Could not open the image: " + imagePath);
    }
    Image orig{.orig_height = (size_t)mat.size().height,
               .orig_width = (size_t)mat.size().width,
               .orig_type = mat.type(),
               .orig_channels = mat.channels()};

    // Ensure the image is in BGR format
    switch (mat.channels()) {
    case 1:
      cv::cvtColor(mat, mat, cv::COLOR_GRAY2BGRA);
      break;
    case 3:
      cv::cvtColor(mat, mat, cv::COLOR_RGB2BGRA);
      break;
    case 4:
      cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
      break;
    default:
      SimpleLogger::LOG_WARN(
          "Non implemented image colors channels processing: ", mat.channels());
      break;
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
      throw ImageHelperException("incorrect image channels");
    }

    // cv::imshow("Original Image step 3", mat);
    // cv::waitKey(1000 * 60 * 2);

    ImageParts imagesParts;
    auto matParts = splitImage(mat, split, withPadding);
    for (auto &matPart : matParts) {
      Image image{.data = matPart,
                  .orig_height = orig.orig_height,
                  .orig_width = orig.orig_width,
                  .orig_type = orig.orig_type,
                  .orig_channels = orig.orig_channels};
      image.resize(resize_x, resize_y);
      auto image_ptr = std::make_shared<Image>(image);
      imagesParts.push_back(image_ptr);
    }

    // Rq. C++ use Return Value Optimization (RVO) to avoid the extra copy or
    // move operation associated with the return.
    return imagesParts;
  } catch (const cv::Exception &e) {
    throw ImageHelperException("Error loading image: " + imagePath + ": " +
                               e.what());
  }
}

ImageParts ImageHelper::generateInputImage(const ImageParts &targetImage,
                                           size_t reduce_factor,
                                           size_t resize_x,
                                           size_t resize_y) const {
  ImageParts imagesParts;
  for (auto &targetPart : targetImage) {
    // clone of the Target image to the Input image
    Image inputImage = {.data = targetPart->data.clone(),
                        .orig_height = targetPart->orig_height,
                        .orig_width = targetPart->orig_width,
                        .orig_type = targetPart->orig_type,
                        .orig_channels = targetPart->orig_channels};

    // reduce the resolution of the input image
    if (reduce_factor != 0) {
      int new_width =
          (int)(inputImage.data.size().width / (float)reduce_factor);
      int new_height =
          (int)(inputImage.data.size().height / (float)reduce_factor);
      inputImage.resize(new_width, new_height);
    }

    // then resize to the layer resolution
    inputImage.resize(resize_x, resize_y);

    // finally convert back to Image
    auto image = std::make_shared<Image>(inputImage);
    imagesParts.push_back(image);
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
  if (imageParts.empty() || split == 0 ||
      (split == 1 && imageParts.size() != 1)) {
    throw ImageHelperException(
        "internal exception: invalid image parts or split number.");
  }
  try {

    auto image = split == 1 ? *imageParts.front()
                            : joinImages(imageParts, (int)split, (int)split);

    if (image.data.empty()) {
      throw ImageHelperException("Image data is empty.");
    }

    image.resize(resize_x, resize_y);

    // convert back the [0,1] float range image to 255 pixel values
    image.data.convertTo(image.data, image.orig_type, 255.0);

    // Convert back to the original color format
    cv::Mat tmp;
    switch (image.orig_channels) {
    case 1:
      cv::cvtColor(image.data, tmp, cv::COLOR_BGRA2GRAY);
      break;
    case 3:
      cv::cvtColor(image.data, tmp, cv::COLOR_BGRA2RGB);
      break;
    case 4:
      cv::cvtColor(image.data, tmp, cv::COLOR_BGRA2RGBA);
      break;
    default:
      SimpleLogger::LOG_WARN(
          "Non implemented image colors channels processing: ",
          image.orig_channels);
      tmp = image.data;
      break;
    }

    // write the image
    // std::vector<int> params;
    // params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    // params.push_back(9); // Compression level
    // if (!cv::imwrite(imagePath, mat, params)) {
    if (!cv::imwrite(imagePath, tmp)) {
      throw ImageHelperException("Error saving image: " + imagePath);
    }
  } catch (ImageHelperException &ihe) {
    throw ihe;
  } catch (const cv::Exception &e) {
    throw ImageHelperException("Error saving image: " + imagePath + ": " +
                               e.what());
  } catch (std::exception &ex) {
    throw ImageHelperException(ex.what());
  }
}

Image ImageHelper::joinImages(const ImageParts &images, int splitsX,
                              int splitsY) const {
  if (images.empty()) {
    throw ImageHelperException("internal exception: empty parts.");
  }
  if (splitsX == 0 || splitsY == 0) {
    throw ImageHelperException("internal exception: split 0.");
  }
  std::vector<cv::Mat> rows;
  for (int i = 0; i < splitsY; ++i) {
    std::vector<cv::Mat> row;
    for (int j = 0; j < splitsX; ++j) {
      row.push_back(images[i * splitsX + j]->data);
    }
    cv::Mat rowImage;
    cv::hconcat(row, rowImage);
    rows.push_back(rowImage);
  }
  cv::Mat result;
  cv::vconcat(rows, result);

  Image image{.data = result,
              .orig_height = images.front()->orig_height,
              .orig_width = images.front()->orig_height,
              .orig_type = images.front()->orig_type,
              .orig_channels = images.front()->orig_channels};
  return image;
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
  size_t numPixels = outputData.total();

  // Calculate the MSE loss
  float mseLoss = 0.0f;
  if (sumSquaredDiff.rows > 0) { // sumSquaredDiff is 1 col, 4 rows (rgba).
    for (int i = 0; i < sumSquaredDiff.rows; i++) {
      mseLoss += static_cast<float>(sumSquaredDiff.val[i]) /
                 static_cast<float>(numPixels);
    }

    mseLoss /= static_cast<float>(sumSquaredDiff.rows);
  }

  return mseLoss;
}