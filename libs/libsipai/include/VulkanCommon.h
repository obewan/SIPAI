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
#include <cstddef>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>
#include <unordered_map>

#if defined(_MSC_VER)
using uint = unsigned int;
#endif

namespace sipai {

inline constexpr const char *cvWindowTitle = "SIPAI";
inline constexpr const int MAX_NEIGHBORS = 4;


// numbers must match the GLSL bindings
enum class EBuffer {
  Parameters = 0,
  InputLayer = 1,
  OutputLayer = 2,
  HiddenLayer1 = 3,
  InputData = 4,
  OutputData = 5,
  OutputLoss = 6,
  SharedOutputValues = 7,
  SharedOutputLoss = 8,
  Vertex = 9,
};

enum class EShader {
  TrainingInit,
  TrainingForward1,
  TrainingForward2,
  TrainingForward3,
  TrainingForward4,
  TrainingBackward1,
  TrainingBackward2,
  TrainingBackward3,
  TrainingBackward4,
  EnhancerForward1,
  EnhancerForward2,
  VertexShader,
  FragmentShader
};

struct ShaderDefinition {
  sipai::EShader name;
  std::string filename;
  std::string templateFilename;
};

const std::list<ShaderDefinition> shader_files {
    { EShader::EnhancerForward1, "data/glsl/EnhancerShader-forward1.comp", "data/glsl/EnhancerShader-forward1.comp.in" },
    { EShader::EnhancerForward2, "data/glsl/EnhancerShader-forward2.comp", "data/glsl/EnhancerShader-forward2.comp.in" },
    { EShader::TrainingInit, "data/glsl/TrainingShader-init.comp", "data/glsl/TrainingShader-init.comp.in" },
    { EShader::TrainingForward1, "data/glsl/TrainingShader-forward1.comp", "data/glsl/TrainingShader-forward1.comp.in" },
    { EShader::TrainingForward2, "data/glsl/TrainingShader-forward2.comp", "data/glsl/TrainingShader-forward2.comp.in" },
    { EShader::TrainingForward3, "data/glsl/TrainingShader-forward3.comp", "data/glsl/TrainingShader-forward3.comp.in" },
    { EShader::TrainingForward4, "data/glsl/TrainingShader-forward4.comp", "data/glsl/TrainingShader-forward4.comp.in" },
    { EShader::TrainingBackward1, "data/glsl/TrainingShader-backward1.comp", "data/glsl/TrainingShader-backward1.comp.in" },
    { EShader::TrainingBackward2, "data/glsl/TrainingShader-backward2.comp", "data/glsl/TrainingShader-backward2.comp.in" },
    { EShader::TrainingBackward3, "data/glsl/TrainingShader-backward3.comp", "data/glsl/TrainingShader-backward3.comp.in" },
    { EShader::TrainingBackward4, "data/glsl/TrainingShader-backward4.comp", "data/glsl/TrainingShader-backward4.comp.in" },
    { EShader::FragmentShader, "data/glsl/FragmentShader.frag", "" },
    { EShader::VertexShader, "data/glsl/VertexShader.vert", "" }
  };

const std::map<EBuffer, std::string, std::less<>> buffer_map{
    {EBuffer::Parameters, "Parameters"},
    {EBuffer::InputLayer, "InputLayer"},
    {EBuffer::OutputLayer, "OutputLayer"},
    {EBuffer::HiddenLayer1, "HiddenLayer1"},
    {EBuffer::InputData, "InputData"},
    {EBuffer::OutputData, "OutputData"},
    {EBuffer::OutputLoss, "OutputLoss"},
    {EBuffer::SharedOutputValues, "SharedOutputValues"},
    {EBuffer::SharedOutputLoss, "SharedOutputLoss"},
    {EBuffer::Vertex, "Vertex"}};

struct Vertex {
  float pos[2];
  float color[3];

  static VkVertexInputBindingDescription getBindingDescription() {
    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescription;
  }

  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions() {
    std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, pos);
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);
    return attributeDescriptions;
  }
};

struct GLSLParameters {
  float learning_rate;
  float error_min;
  float error_max;
};

struct GLSLNeighbor {
  bool is_used;
  uint index_x;
  uint index_y;
  std::vector<float> weight;
};

struct GLSLNeuron {
  uint index_x;
  uint index_y;
  std::vector<std::vector<std::vector<float>>> weights;
  GLSLNeighbor neighbors[MAX_NEIGHBORS];
};

struct GLSLInputData {
  std::vector<std::vector<cv::Vec4f>> inputValues;
  std::vector<std::vector<cv::Vec4f>> targetValues;
  bool is_validation;
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
  std::vector<std::vector<std::vector<float>>> values;
  std::vector<std::vector<std::vector<float>>> errors;
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
  std::unique_ptr<std::vector<uint32_t>> shader = nullptr;
  VkShaderModule module = VK_NULL_HANDLE;
  bool isReady = false;
};

struct Vulkan {
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDevice logicalDevice = VK_NULL_HANDLE;
  VkFence computeFence = VK_NULL_HANDLE;
  VkFence inFlightFence = VK_NULL_HANDLE;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkQueue queueCompute = VK_NULL_HANDLE;
  VkQueue queueGraphics = VK_NULL_HANDLE;
  VkRenderPass renderPass = VK_NULL_HANDLE;
  VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
  VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapChain = VK_NULL_HANDLE;
  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;
  std::list<VkComputePipelineCreateInfo> computePipelineInfos;
  VkGraphicsPipelineCreateInfo graphicPipelineInfo = {};
  VkPipeline graphicPipeline = VK_NULL_HANDLE;
  std::unordered_map<EShader, VkPipeline> computePipelines;
  std::vector<Vertex> vertices;
  std::vector<Buffer> buffers;
  std::vector<Shader> shaders;
  std::vector<VkCommandBuffer> commandBufferPool;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkImage> swapChainImages;
  std::vector<VkImageView> swapChainImageViews;
  unsigned int queueComputeIndex = 0;
  unsigned int queueGraphicsIndex = 0;
  unsigned int window_width = 800;
  unsigned int window_height = 600;
  bool isInitialized = false;
  size_t maxSizeX = 0;
  size_t maxSizeY = 0;
};

} // namespace sipai