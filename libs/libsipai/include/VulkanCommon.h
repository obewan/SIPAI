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
#include <opencv2/opencv.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>

#if defined(_MSC_VER)
using uint = unsigned int;
#endif

namespace sipai {

// numbers must match the GLSL bindings
enum class EBuffer {
  Parameters = 0,
  Data = 1,
  InputLayer = 2,
  OutputLayer = 3,
  HiddenLayer1 = 4,
};

enum class EShader {
  TrainingMonitored,
};

const std::map<EBuffer, std::string, std::less<>> buffer_map{
    {EBuffer::Parameters, "Parameters"},
    {EBuffer::Data, "Data"},
    {EBuffer::InputLayer, "InputLayer"},
    {EBuffer::OutputLayer, "OutputLayer"},
    {EBuffer::HiddenLayer1, "HiddenLayer1"}};

struct GLSLNeighbor {
  bool is_used;
  uint index_x;
  uint index_y;
  cv::Vec4f weight;
};

struct GLSLNeuron {
  uint index_x;
  uint index_y;
  cv::Vec4f **weights;
  GLSLNeighbor neighbors[4];
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