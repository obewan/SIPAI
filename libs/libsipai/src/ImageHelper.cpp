#include "ImageHelper.h"
#include "exception/ImageHelperException.h"
#include <opencv2/imgcodecs.hpp>

using namespace sipai;

void ImageHelper::loadImageIntoMemoryVK(const VkPhysicalDevice &physicalDevice,
                                        const VkDevice &device,
                                        const std::string &imagePath,
                                        size_t split, bool withPadding,
                                        size_t resize_x,
                                        size_t resize_y) const {
  // Load image using OpenCV
  cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

  // Convert image data to RGBA (if not already)
  cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);

  // Create a Vulkan buffer and copy the image data into it
  VkDeviceSize imageSize = img.total() * img.elemSize();
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(physicalDevice, device, imageSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  // Map the buffer memory and copy the image data
  void *data;
  vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, img.data, (size_t)imageSize);
  vkUnmapMemory(device, stagingBufferMemory);
}

void ImageHelper::createBuffer(const VkPhysicalDevice &physicalDevice,
                               const VkDevice &device, VkDeviceSize size,
                               VkBufferUsageFlags usage,
                               VkMemoryPropertyFlags properties,
                               VkBuffer &buffer,
                               VkDeviceMemory &bufferMemory) const {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(
      physicalDevice, memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

uint32_t ImageHelper::findMemoryType(const VkPhysicalDevice &physicalDevice,
                                     uint32_t typeFilter,
                                     VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

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
      cv::resize(matPart, matPart, cv::Size((int)resize_x, (int)resize_y));
    } else {
      resize_x = s.width;
      resize_y = s.height;
    }

    const auto &data = convertToRGBAVector(matPart);
    auto image =
        std::make_unique<Image>(data, resize_x, resize_y, s.width, s.height);
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
  for (auto &imagePart : targetImage) {
    auto mat = convertToMat(*imagePart);

    // reduce the resolution of the input image
    cv::Size s = mat.size();
    if (reduce_factor != 0) {
      int new_width = (int)(s.width * (1.0f / (float)reduce_factor));
      int new_height = (int)(s.height * (1.0f / (float)reduce_factor));
      cv::resize(mat, mat, cv::Size(new_width, new_height));
    }
    // get the new size if any resize
    s = mat.size();

    // then resize to the layer resolution
    if (resize_x > 0 && resize_y > 0) {
      cv::resize(mat, mat, cv::Size((int)resize_x, (int)resize_y));
    } else {
      resize_x = s.width;
      resize_y = s.height;
    }

    // finally convert back to Image
    const auto &data = convertToRGBAVector(mat);
    auto image =
        std::make_unique<Image>(data, resize_x, resize_y, s.width, s.height);
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
    std::vector<cv::Mat> mats(imageParts.size());
    std::transform(imageParts.begin(), imageParts.end(), mats.begin(),
                   [&](const std::unique_ptr<Image> &part) {
                     return convertToMat(*part);
                   });
    auto mat = joinImages(mats, (int)split, (int)split);

    if (resize_x > 0 && resize_y > 0) {
      cv::resize(mat, mat, cv::Size((int)resize_x, (int)resize_y));
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
  cv::Mat dest((int)image.size_y, (int)image.size_x, CV_8UC4);
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