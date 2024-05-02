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
#include "exception/VulkanControllerException.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <map>

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


  // number must match the binding number
  enum class EBuffer {
      CurrentLayerNeurons = 0,
      CurrentLayerValues = 1,
      CurrentNeighborsErrors = 2,
      CurrentNeighborsWeights = 3,
      AdjacentLayerNeurons = 4,
      AdjacentLayerValues = 5,
      LayerWeights = 6,
      Parameters = 7,
      Output = 8
  };

  const std::map<EBuffer, std::string, std::less<>> buffer_map{
      { EBuffer::CurrentLayerNeurons,"CurrentLayerNeurons"},
      { EBuffer::CurrentLayerValues, "CurrentLayerValues"},
      { EBuffer::CurrentNeighborsErrors, "CurrentNeighborsErrors"},
      { EBuffer::CurrentNeighborsWeights, "CurrentNeighborsWeights"},
      { EBuffer::AdjacentLayerNeurons, "AdjacentLayerNeurons"},
      { EBuffer::AdjacentLayerValues, "AdjacentLayerValues"},
      { EBuffer::LayerWeights, "LayerWeights"},
      { EBuffer::Parameters, "Parameters"},
      { EBuffer::Output, "Output"} };


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

  struct Buffer {
      EBuffer name;
      uint binding = 0;
      VkBuffer buffer = VK_NULL_HANDLE;
      VkDeviceMemory memory = VK_NULL_HANDLE;
      VkBufferCreateInfo info{};
      void* data = nullptr;
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

  Buffer& getBuffer(const EBuffer &bufferName) {
      auto it = std::find_if(buffers_.begin(), buffers_.end(), [&bufferName](auto& buffer) {
          return buffer.name == bufferName;
          });
      if (it != buffers_.end()) {
          return *it;
      }
      else {
          throw VulkanControllerException("buffer not found");
      }
  }

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

  void _copyNeuronsToBuffer(const NeuronMat &neurons, Buffer& buffer);
  void _copyMatToBuffer(const cv::Mat &mat, Buffer& buffer);
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
  void _createBuffers();
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

  std::vector<Buffer> buffers_;
  std::vector<VkCommandBuffer> commandBufferPool_;
};
} // namespace sipai