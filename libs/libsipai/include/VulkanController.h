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

  struct GLSLNeuron {
    uint index_x;
    uint index_y;
    uint weightsIndex;
    uint neighborsIndex;
    uint neighborsSize;
  };

  struct GLSLParameters {
    float error_min;
    float error_max;
    float activationAlpha;
    uint currentLayerSizeX;
    uint currentLayerSizeY;
    uint previousLayerSizeX;
    uint previousLayerSizeY;
    uint nextLayerSizeX;
    uint nextLayerSizeY;
    uint activationFunction;
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

  uint32_t _findMemoryType(uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) const;

  VkCommandBuffer _beginSingleTimeCommands();
  void _endSingleTimeCommands(VkCommandBuffer &commandBuffer);

  std::unique_ptr<std::vector<uint32_t>> _loadShader(const std::string &path);
  void _computeShader(const NeuronMat &neurons, VkCommandBuffer &commandBuffer,
                      VkPipeline &pipeline);

  void _copyNeuronsToBuffer(const NeuronMat &neurons,
                            VkBufferCreateInfo &bufferInfo, void *&bufferData);
  void _copyMatToBuffer(const cv::Mat &mat, VkBufferCreateInfo &bufferInfo,
                        void *&bufferData);
  void _copyOutputBufferToMat(cv::Mat &mat);
  void _copyParametersToParametersBuffer(Layer *currentLayer);
  void _copyNeuronsWeightsToWeightsBuffer(const NeuronMat &neurons);
  void _copyNeuronNeighboorsConnectionToBuffer(Layer *layer);
  void _copyNeuronNeighboorsIndexesToBuffer(const NeuronMat &neurons);

  void _createCommandPool();
  void _createCommandBufferPool();
  void _createPipelineLayout();
  void _createDescriptorSet();
  void _createDescriptorSetLayout();
  void _createFence();
  void _createDescriptorPool();
  void _createBuffers(size_t max_size);
  void _createBuffer(VkDeviceSize size, VkBufferCreateInfo &bufferInfo,
                     VkBuffer &buffer, VkDeviceMemory &bufferMemory);
  void _createDataMapping();
  void _createShaderModules();
  void _createShadersComputePipelines();
  void _bindBuffers();

  std::optional<unsigned int> _pickQueueFamily();
  std::optional<VkPhysicalDevice> _pickPhysicalDevice();

  std::unique_ptr<std::vector<uint32_t>> forwardShader_;
  std::unique_ptr<std::vector<uint32_t>> backwardShader_;
  VkShaderModule forwardShaderModule_ = VK_NULL_HANDLE;
  VkShaderModule backwardShaderModule_ = VK_NULL_HANDLE;
  VkPipeline forwardComputePipeline_ = VK_NULL_HANDLE;
  VkPipeline backwardComputePipeline_ = VK_NULL_HANDLE;
  VkComputePipelineCreateInfo forwardPipelineInfo_{};
  VkComputePipelineCreateInfo backwardPipelineInfo_{};

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

  // Binding 0
  VkBuffer currentLayerBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory currentLayerBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo currentLayerBufferInfo_{};
  void *currentLayerData_ = nullptr;

  // Binding 1
  VkBuffer currentLayerValuesBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory currentLayerValuesBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo currentLayerValuesBufferInfo_{};
  void *currentLayerValuesData_ = nullptr;

  // Binding 2
  VkBuffer currentNeighborsErrorsBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory currentNeighborsErrorsBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo currentNeighborsErrorsBufferInfo_{};
  void *currentNeighborsErrorsData_ = nullptr;

  // Binding 4
  VkBuffer currentNeighborsWeightsBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory currentNeighborsWeightsBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo currentNeighborsWeightsBufferInfo_{};
  void *currentNeighborsWeightsData_ = nullptr;

  // Binding 5
  VkBuffer adjacentLayerBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory adjacentLayerBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo adjacentLayerBufferInfo_{};
  void *adjacentLayerData_ = nullptr;

  // Binding 6
  VkBuffer adjacentLayerValuesBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory adjacentLayerValuesBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo adjacentLayerValuesBufferInfo_{};
  void *adjacentLayerValuesData_ = nullptr;

  // Binding 7
  VkBuffer weightsBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory weightsBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo weightsBufferInfo_{};
  void *weightsData_ = nullptr;

  // Binding 8
  VkBuffer parametersBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory parametersBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo parametersBufferInfo_{};
  void *parametersData_ = nullptr;

  // Binding 9
  VkBuffer outputBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory outputBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo outputBufferInfo_{};
  void *outputData_ = nullptr;

  std::vector<VkCommandBuffer> commandBufferPool_;
};
} // namespace sipai