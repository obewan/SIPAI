/**
 * @file VulkanCommon.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan Common
 * @date 2024-05-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include <map>
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace sipai {

// numbers must match the GLSL bindings
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

enum class EShader {
  Forward,
  Backward,
};

const std::map<EBuffer, std::string, std::less<>> buffer_map{
    {EBuffer::CurrentLayerNeurons, "CurrentLayerNeurons"},
    {EBuffer::CurrentLayerValues, "CurrentLayerValues"},
    {EBuffer::CurrentNeighborsErrors, "CurrentNeighborsErrors"},
    {EBuffer::CurrentNeighborsWeights, "CurrentNeighborsWeights"},
    {EBuffer::AdjacentLayerNeurons, "AdjacentLayerNeurons"},
    {EBuffer::AdjacentLayerValues, "AdjacentLayerValues"},
    {EBuffer::LayerWeights, "LayerWeights"},
    {EBuffer::Parameters, "Parameters"},
    {EBuffer::Output, "Output"}};

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
  void *data = nullptr;
};

struct Shader {
  EShader shadername;
  std::string filename;
  std::unique_ptr<std::vector<uint32_t>> shader;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkComputePipelineCreateInfo info{};
};

struct Vulkan {
  VkInstance vkInstance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice logicalDevice = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkFence computeFence = VK_NULL_HANDLE;
  std::vector<Shader> shaders;
  std::vector<Buffer> buffers;
  std::vector<VkCommandBuffer> commandBufferPool;
  unsigned int queueFamilyIndex = 0;
  bool isInitialized = false;
};

} // namespace sipai