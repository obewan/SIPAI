/**
 * @file VulkanControllerTest.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief For testing purposes
 * @date 2024-05-18
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "Layer.h"
#include "VulkanBuilder.h"
#include "VulkanCommon.h"
#include "VulkanHelper.h"
#include "exception/VulkanControllerException.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace sipai {
class VulkanControllerTest {
public:
  static VulkanControllerTest &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { controllerInstance_.reset(new VulkanControllerTest); });
    return *controllerInstance_;
  }
  static const VulkanControllerTest &getConstInstance() {
    return const_cast<const VulkanControllerTest &>(getInstance());
  }
  VulkanControllerTest(VulkanControllerTest const &) = delete;
  void operator=(VulkanControllerTest const &) = delete;
  ~VulkanControllerTest() { destroy(); }

  struct ResultTest2 {
    Layer *layer;
    float loss;
  };

  /**
   * @brief Helper function to copy data from the buffer
   *
   * @tparam T
   * @param bufferData
   * @param offset
   * @return T
   */
  template <typename T>
  T getDataFromBuffer(const void *bufferData, uint32_t &offset) {
    const T *typedBufferData = reinterpret_cast<const T *>(bufferData);
    T value = *(typedBufferData + offset / sizeof(T));
    offset += sizeof(T);
    return value;
  }

  /**
   * @brief Set the Data To Buffer
   *
   * @tparam T
   * @param buffer
   * @param data
   * @return size_t buffer new position
   */
  template <typename T> uint8_t *copyToBuffer(uint8_t *buffer, const T &data) {
    memcpy(buffer, &data, sizeof(T));
    return buffer + sizeof(T);
  }

  float test1();
  ResultTest2 test2();

  bool initialize();

  const bool IsInitialized() { return vulkan_->isInitialized; }

  void destroy() { builder_.clear(); };

  VkDevice &getDevice() { return vulkan_->logicalDevice; }

  Buffer &getBuffer(const EBuffer &bufferName) {
    if (!vulkan_) {
      throw VulkanControllerException("vulkan is null pointer.");
    }
    auto it = std::find_if(
        vulkan_->buffers.begin(), vulkan_->buffers.end(),
        [&bufferName](auto &buffer) { return buffer.name == bufferName; });
    if (it != vulkan_->buffers.end()) {
      return *it;
    } else {
      throw VulkanControllerException("buffer not found.");
    }
  }

  Shader &getShader(const EShader &shaderName) {
    if (!vulkan_) {
      throw VulkanControllerException("vulkan is null pointer.");
    }
    auto it = std::find_if(vulkan_->shaders.begin(), vulkan_->shaders.end(),
                           [&shaderName](auto &shader) {
                             return shader.shadername == shaderName;
                           });
    if (it != vulkan_->shaders.end()) {
      return *it;
    } else {
      throw VulkanControllerException("shader not found.");
    }
  }

  std::shared_ptr<Vulkan> getVulkan() { return vulkan_; }

  std::string testFile1 = "../../test/data/shaders/shader_test1.comp";
  std::string testFile2 = "../../test/data/shaders/shader_test2.comp";

private:
  VulkanControllerTest() {
    vulkan_ = std::make_shared<Vulkan>();
    helper_.setVulkan(vulkan_);
  };
  static std::unique_ptr<VulkanControllerTest> controllerInstance_;

  void _computeShader(VkPipeline &pipeline);

  void _copyParameters();
  void _copyInputLayer();
  void _copyOutputLayer();
  void _copyHiddenLayer1();
  void _copyInputData(const cv::Mat &inputValues, const cv::Mat &targetValues,
                      bool is_validation);
  float _readOutputLoss();
  Layer *_getHiddenLayer1();

  std::shared_ptr<Vulkan> vulkan_;
  VulkanBuilder builder_;
  VulkanHelper helper_;
};
} // namespace sipai