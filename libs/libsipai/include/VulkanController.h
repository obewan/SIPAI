/**
 * @file VulkanController.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan Controller
 * @date 2024-04-19
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
class VulkanController {
public:
  static VulkanController &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { controllerInstance_.reset(new VulkanController); });
    return *controllerInstance_;
  }
  static const VulkanController &getConstInstance() {
    return const_cast<const VulkanController &>(getInstance());
  }
  VulkanController(VulkanController const &) = delete;
  void operator=(VulkanController const &) = delete;
  ~VulkanController() { destroy(); }

  bool initialize(bool enableDebug = false);

  const bool IsInitialized() { return vulkan_->isInitialized; }

  /**
   * @brief Vulkan training or validation on an input image
   *
   *  @return float computed loss between the generated output and the expected
   * images after the training or the validation
   */
  float trainingMonitored(const std::shared_ptr<sipai::Image> &inputValues,
                          const std::shared_ptr<sipai::Image> &targetValues,
                          const TrainingPhase &phase);

  /**
   * @brief Update the Neural Network with values from Vulkan.
   *
   */
  void updateNeuralNetwork();

  /**
   * @brief Destroy the device instance, cleaning ressources
   *
   */
  void destroy() { builder_.clear(); };

  /**
   * @brief Get the Logical Device
   *
   * @return VkDevice&
   */
  VkDevice &getDevice() { return vulkan_->logicalDevice; }

  /**
   * @brief Get a Buffer
   *
   * @param bufferName
   * @return Buffer&
   */
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

  /**
   * @brief Helper function to copy data from the buffer with proper alignment
   * and endianness
   *
   * @tparam T
   * @param bufferData
   * @param offset
   * @return T
   */
  template <typename T>
  T getValueFromBuffer(const uint8_t *bufferData, uint32_t &offset) {
    // Calculate the alignment for the type T
    constexpr size_t alignment = alignof(T);
    // Calculate the next aligned address
    const uintptr_t alignedAddress =
        (reinterpret_cast<uintptr_t>(bufferData + offset) + alignment - 1) &
        ~(alignment - 1);
    // Update the offset to the next aligned address
    offset = static_cast<uint32_t>(alignedAddress -
                                   reinterpret_cast<uintptr_t>(bufferData));
    // Cast the aligned address back to the pointer of type T
    const T *alignedPtr = reinterpret_cast<const T *>(alignedAddress);
    // Increment the offset by the size of T
    offset += sizeof(T);
    // Return the value
    return *alignedPtr;
  }

private:
  VulkanController() {
    vulkan_ = std::make_shared<Vulkan>();
    helper_.setVulkan(vulkan_);
  };
  static std::unique_ptr<VulkanController> controllerInstance_;

  void _computeShader(VkPipeline &pipeline);
  void _readBackHiddenLayer1();
  void _readBackOutputLayer();

  void _copyParameters();
  void _copyInputLayer();
  void _copyOutputLayer();
  void _copyHiddenLayer1();
  void _copyInputData(const cv::Mat &inputValues, const cv::Mat &targetValues,
                      bool is_validation);
  std::unique_ptr<GLSLOutputData> _getOutputData();

  std::shared_ptr<Vulkan> vulkan_;
  VulkanBuilder builder_;
  VulkanHelper helper_;
};
} // namespace sipai