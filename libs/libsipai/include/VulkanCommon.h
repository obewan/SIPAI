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

const int MAX_NEIGHBORS = 4;

// numbers must match the GLSL bindings
enum class EBuffer {
  Parameters = 0,
  InputLayer = 1,
  OutputLayer = 2,
  HiddenLayer1 = 3,
  InputData = 4,
  OutputData = 5,
  OutputLoss = 6
};

enum class EShader {
  TrainingMonitored,

  // For Testing
  Test1
};

const std::map<EBuffer, std::string, std::less<>> buffer_map{
    {EBuffer::Parameters, "Parameters"},
    {EBuffer::InputLayer, "InputLayer"},
    {EBuffer::OutputLayer, "OutputLayer"},
    {EBuffer::HiddenLayer1, "HiddenLayer1"},
    {EBuffer::InputData, "InputData"},
    {EBuffer::OutputData, "OutputData"},
    {EBuffer::OutputLoss, "OutputLoss"}};

struct GLSLParameters {
  float learning_rate;
  float error_min;
  float error_max;
};

struct GLSLNeighbor {
  bool is_used;
  uint index_x;
  uint index_y;
  cv::Vec4f weight;
};

struct GLSLNeuron {
  uint index_x;
  uint index_y;
  std::vector<std::vector<cv::Vec4f>> weights;
  GLSLNeighbor neighbors[MAX_NEIGHBORS];
};

struct GLSLInputData {
  std::vector<std::vector<cv::Vec4f>> inputValues;
  std::vector<std::vector<cv::Vec4f>> targetValues;
  bool is_validation;
};

// special format after transformations and merge
struct GLSLOutputData {
  cv::Mat outputValues;
  float loss;
};

struct GLSLInputLayer {
  float activation_alpha;
  uint activation_function;
  uint size_x;
  uint size_y;
};

struct GLSLOutputLayer {
  std::vector<std::vector<GLSLNeuron>> neurons;
  std::vector<std::vector<cv::Vec4f>> errors;
  float activation_alpha;
  uint activation_function;
  uint size_x;
  uint size_y;
};

struct GLSLHiddenLayer {
  std::vector<std::vector<GLSLNeuron>> neurons;
  std::vector<std::vector<cv::Vec4f>> values;
  std::vector<std::vector<cv::Vec4f>> errors;
  float activation_alpha;
  uint activation_function;
  uint size_x;
  uint size_y;
};

struct Buffer {
  EBuffer name;
  uint binding = 0;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBufferCreateInfo info = {};
  void *data = nullptr;
  bool isMemoryMapped = false;
};

struct Shader {
  EShader shadername;
  std::string filename;
  std::unique_ptr<std::vector<uint32_t>> shader;
  VkShaderModule module = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkComputePipelineCreateInfo info = {};
  bool isReady = false;
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