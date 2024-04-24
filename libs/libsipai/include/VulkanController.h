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
#include <atomic>
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

  struct GLSLActivationFunction {
    int value;
    float alpha;
  };

  void initialize();

  const bool IsInitialized() { return isInitialized_.load(); }

  /**
   * @brief Vulkan Forward Propagation
   *
   * @param previousLayer
   * @param currentLayer
   */
  void forwardPropagation(Layer *previousLayer, Layer *currentLayer);

  /**
   * @brief Destroy the device instance, cleaning ressources
   *
   */
  void destroy();

  /**
   * @brief Get the Logical Device
   *
   * @return VkDevice&
   */
  VkDevice &getDevice() { return logicalDevice_; }

private:
  VulkanController() = default;
  static std::unique_ptr<VulkanController> controllerInstance_;

  std::unique_ptr<std::vector<uint32_t>> forwardShader_;

  std::atomic<bool> isInitialized_ = false;
  unsigned int queueFamilyIndex_ = 0;

  VkInstance vkInstance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice logicalDevice_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkFence computeFence_ = VK_NULL_HANDLE;

  VkBuffer inputBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory inputBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo inputBufferInfo_{};
  void *inputData_ = nullptr;

  VkBuffer outputBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory outputBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo outputBufferInfo_{};
  void *outputData_ = nullptr;

  VkBuffer currentBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory currentBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo currentBufferInfo_{};
  void *currentData_ = nullptr;

  VkBuffer activationFunctionBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory activationFunctionBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo activationFunctionBufferInfo_{};
  void *activationFunctionData_ = nullptr;

  VkBuffer weightsBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory weightsBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo weightsBufferInfo_{};
  void *weightsData_ = nullptr;

  std::vector<VkCommandBuffer> commandBufferPool_;
  VkShaderModule forwardShaderModule_ = VK_NULL_HANDLE;
  VkPipeline forwardComputePipeline_ = VK_NULL_HANDLE;

  uint32_t _findMemoryType(uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) const;

  VkCommandBuffer _beginSingleTimeCommands();
  void _endSingleTimeCommands(VkCommandBuffer commandBuffer);

  std::unique_ptr<std::vector<uint32_t>> _loadShader(const std::string &path);
  void _computeShader(std::vector<Neuron> &neurons);

  void _copyNeuronsDataToInputBuffer(const std::vector<Neuron> &neurons);
  void _copyNeuronsDataToCurrentBuffer(const std::vector<Neuron> &neurons);
  void _copyOutputBufferToNeuronsData(std::vector<Neuron> &neurons);
  void _copyActivationFunctionToActivationFunctionBuffer(
      const EActivationFunction &activationFunction, float alpha);
  void _copyNeuronsWeightsToWeightsBuffer(const std::vector<Neuron> &neurons);

  void _createCommandPool();
  void _createCommandBufferPool();
  void _createPipelineLayout();
  void _createDescriptorSet();
  void _createDescriptorSetLayout();
  void _createFence();
  void _createDescriptorPool(size_t max_size);
  void _createBuffers(size_t max_size);
  void _createBuffer(VkDeviceSize size, VkBufferCreateInfo &bufferInfo,
                     VkBuffer &buffer, VkDeviceMemory &bufferMemory);
  void _createDataMapping();
  void _createForwardShaderModule();
  void _createForwardComputePipeline();
  void _bindBuffers();

  std::optional<unsigned int> _pickQueueFamily();
  std::optional<VkPhysicalDevice> _pickPhysicalDevice();
};
} // namespace sipai