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
  float trainingMonitored(const TrainingPhase &phase);

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

private:
  VulkanController() {
    vulkan_ = std::make_shared<Vulkan>();
    helper_.setVulkan(vulkan_);
  };
  static std::unique_ptr<VulkanController> controllerInstance_;

  void _computeShader(VkPipeline &pipeline);

  void _copyParameters();
  void _copyInputLayer();
  void _copyOutputLayer();
  void _copyHiddenLayer1();
  void _copyInputData();

  std::shared_ptr<Vulkan> vulkan_;
  VulkanBuilder builder_;
  VulkanHelper helper_;
};
} // namespace sipai