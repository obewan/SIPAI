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
#include "Neuron.h"
#include "VulkanBuilder.h"
#include "VulkanCommon.h"
#include "exception/VulkanControllerException.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

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

  void initialize(bool enableDebug = false);

  const bool IsInitialized() { return vulkan_->isInitialized; }

  /**
   * @brief Vulkan Forward Propagation
   *
   * @param previousLayer
   * @param currentLayer
   */
  void forwardPropagation(Layer *previousLayer, Layer *currentLayer);

  /**
   * @brief Vulkan Backward Propagation
   *
   * @param nextLayer
   * @param currentLayer
   */
  void backwardPropagation(Layer *nextLayer, Layer *currentLayer);

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
  VulkanController() { vulkan_ = std::make_shared<Vulkan>(); };
  static std::unique_ptr<VulkanController> controllerInstance_;

  VkCommandBuffer _beginSingleTimeCommands();
  void _endSingleTimeCommands(VkCommandBuffer &commandBuffer);

  void _computeShader(const NeuronMat &neurons, VkCommandBuffer &commandBuffer,
                      VkPipeline &pipeline);

  void _copyNeuronsToBuffer(const NeuronMat &neurons, Buffer &buffer);
  void _copyMatToBuffer(const cv::Mat &mat, Buffer &buffer);
  void _copyOutputBufferToMat(cv::Mat &mat);
  void _copyParametersToParametersBuffer(Layer *currentLayer);
  void _copyNeuronsWeightsToWeightsBuffer(const NeuronMat &neurons);
  void _copyNeuronNeighboorsConnectionToBuffer(Layer *layer);
  void _copyNeuronNeighboorsIndexesToBuffer(const NeuronMat &neurons);

  std::shared_ptr<Vulkan> vulkan_;
  VulkanBuilder builder_;
};
} // namespace sipai