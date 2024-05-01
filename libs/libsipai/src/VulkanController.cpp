#include "VulkanController.h"
#include "ActivationFunctions.h"
#include "Manager.h"
#include "Neuron.h"
#include "SimpleLogger.h"
#include "exception/VulkanControllerException.h"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

using namespace sipai;

constexpr size_t BUFFER_COUNT = 10;
constexpr size_t COMMAND_POOL_SIZE = 10;
constexpr size_t MAX_NEIGHBOORS_PER_NEURON = 4;

std::unique_ptr<VulkanController> VulkanController::controllerInstance_ =
    nullptr;

void VulkanController::initialize() {
  if (isInitialized_) {
    return;
  }

  const auto &manager = Manager::getConstInstance();
  if (!manager.network) {
    return;
  }

  // Initialize Vulkan
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "SIPAI";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfoInstance{};
  createInfoInstance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfoInstance.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfoInstance, nullptr, &vkInstance_) !=
      VK_SUCCESS) {
    throw VulkanControllerException("failed to create instance!");
  }

  // Create a device
  auto physicalDevice = _pickPhysicalDevice();
  if (!physicalDevice.has_value()) {
    throw VulkanControllerException("failed to find a suitable GPU!");
  }
  physicalDevice_ = physicalDevice.value();

  auto queueFamilyIndex = _pickQueueFamily();
  if (!queueFamilyIndex.has_value()) {
    throw VulkanControllerException(
        "failed to find GPUs with Vulkan queue support!");
  }
  queueFamilyIndex_ = queueFamilyIndex.value();

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};
  // Commented as not required
  // deviceFeatures.logicOp = VK_TRUE; // Enable logical operation feature
  // deviceFeatures.shaderFloat64 = VK_TRUE; // Enable 64-bit floats in shader

  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = &queueCreateInfo;
  createInfoDevice.queueCreateInfoCount = 1;
  createInfoDevice.pEnabledFeatures = &deviceFeatures;

  if (vkCreateDevice(physicalDevice_, &createInfoDevice, nullptr,
                     &logicalDevice_) != VK_SUCCESS) {
    throw VulkanControllerException("failed to create logical device!");
  }
  vkGetDeviceQueue(logicalDevice_, queueFamilyIndex_, 0, &queue_);

  forwardShader_ = _loadShader(manager.app_params.forwardShader);
  backwardShader_ = _loadShader(manager.app_params.backwardShader);

  size_t max_size = manager.network->max_neurons();
  _createCommandPool();
  _createCommandBufferPool();
  _createDescriptorSetLayout();
  _createDescriptorPool(max_size);
  _createDescriptorSet();
  _createPipelineLayout();
  _createFence();
  _createBuffers(max_size);
  _createDataMapping();
  _createShaderModules();
  _createShadersComputePipelines();
  _bindBuffers();

  isInitialized_ = true;
}

void VulkanController::forwardPropagation(Layer *previousLayer,
                                          Layer *currentLayer) {
  if (!IsInitialized()) {
    throw NeuralNetworkException("Vulkan controller is not initialized.");
  }

  // Prepare data for the shader
  auto commandBuffer = _beginSingleTimeCommands();
  _copyNeuronsWeightsToWeightsBuffer(
      currentLayer->neurons); // before others for weights index
  _endSingleTimeCommands(commandBuffer);

  commandBuffer = _beginSingleTimeCommands();
  _copyNeuronsToBuffer(previousLayer->neurons, adjacentLayerBufferInfo_,
                       adjacentLayerData_);
  _copyMatToBuffer(previousLayer->values, adjacentLayerValuesBufferInfo_,
                   adjacentLayerValuesData_);
  _copyNeuronsToBuffer(currentLayer->neurons, currentLayerBufferInfo_,
                       currentLayerData_);
  _copyParametersToParametersBuffer(currentLayer);
  _endSingleTimeCommands(commandBuffer);

  // Run the shader
  _computeShader(currentLayer->neurons, forwardComputePipeline_);

  // Get the results
  commandBuffer = _beginSingleTimeCommands();
  _copyOutputBufferToMat(currentLayer->values);
  _endSingleTimeCommands(commandBuffer);
}

void VulkanController::backwardPropagation(Layer *nextLayer,
                                           Layer *currentLayer) {
  if (!IsInitialized()) {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }
}

std::unique_ptr<std::vector<uint32_t>>
VulkanController::_loadShader(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    throw VulkanControllerException("GLSL file does not exist: " + path);
  }
  // Use glslangValidator to compile the GLSL shader to SPIR-V
  std::stringstream sst;
#ifdef _WIN32
  sst << "glslangValidator.exe -V " << path << " -o shader.spv";
#else
  sst << "glslangValidator -V " << path << " -o shader.spv";
#endif
  system(sst.str().c_str());

  // Load the compiled SPIR-V into a std::vector<uint32_t>
  std::ifstream file("shader.spv", std::ios::binary | std::ios::ate);
  if (!file.good()) {
    throw VulkanControllerException("Failed to open SPIR-V file");
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto compiledShaderCode =
      std::make_unique<std::vector<uint32_t>>(size / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char *>(compiledShaderCode->data()), size)) {
    throw VulkanControllerException("Failed to read SPIR-V file");
  }
  return compiledShaderCode;
}

void VulkanController::_computeShader(const NeuronMat &neurons,
                                      VkPipeline pipeline) {
  VkCommandBuffer commandBuffer = _beginSingleTimeCommands();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
  uint32_t rows = static_cast<uint32_t>(neurons.size());
  uint32_t cols = rows > 0 ? static_cast<uint32_t>(neurons[0].size()) : 0;
  vkCmdDispatch(commandBuffer, cols, rows, 1);
  _endSingleTimeCommands(commandBuffer);
}

VkCommandBuffer VulkanController::_beginSingleTimeCommands() {
  // Take a command buffer from the pool
  VkCommandBuffer commandBuffer = commandBufferPool_.back();
  commandBufferPool_.pop_back();

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  // Starts recording the command
  auto result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan command buffer start error.");
  }
  return commandBuffer;
}

void VulkanController::_endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  // Ends recording the command
  auto result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan command buffer end error.");
  }
  // Reset the fence
  result = vkResetFences(logicalDevice_, 1, &computeFence_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan reset fence error.");
  }
  // Submit the command to the queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  result = vkQueueSubmit(queue_, 1, &submitInfo, computeFence_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan queue submit error.");
  }
  // Wait for the fence to signal that the GPU has finished
  result =
      vkWaitForFences(logicalDevice_, 1, &computeFence_, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan wait for fence error.");
  }

  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan reset command buffer error.");
  }

  commandBufferPool_.push_back(commandBuffer);
}

void VulkanController::_createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = queueFamilyIndex_;
  poolInfo.flags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional flags
  if (vkCreateCommandPool(logicalDevice_, &poolInfo, nullptr, &commandPool_) !=
      VK_SUCCESS) {
    throw VulkanControllerException("Failed to create command pool!");
  }
}

void VulkanController::_createCommandBufferPool() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool_;
  allocInfo.commandBufferCount = COMMAND_POOL_SIZE;
  commandBufferPool_ = std::vector<VkCommandBuffer>(COMMAND_POOL_SIZE);
  if (vkAllocateCommandBuffers(logicalDevice_, &allocInfo,
                               commandBufferPool_.data()) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to allocate command buffers!");
  }
}

void VulkanController::_createDescriptorSetLayout() {
  std::array<VkDescriptorSetLayoutBinding, BUFFER_COUNT> layoutBindings{};
  // Buffer layout binding
  for (size_t i = 0; i < layoutBindings.size(); i++) {
    layoutBindings[i].binding = (unsigned int)i; // binding number
    layoutBindings[i].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // type of the bound descriptor(s)
    layoutBindings[i].descriptorCount =
        1; // number of descriptors in the binding
    layoutBindings[i].stageFlags =
        VK_SHADER_STAGE_COMPUTE_BIT; // shader stages that can access the
                                     // binding
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(
      layoutBindings.size()); // number of bindings in the descriptor set
  layoutInfo.pBindings = layoutBindings.data(); // array of bindings
  if (vkCreateDescriptorSetLayout(logicalDevice_, &layoutInfo, nullptr,
                                  &descriptorSetLayout_) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create descriptor set layout!");
  }
}

void VulkanController::_createDescriptorPool(size_t max_size) {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(max_size);
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = static_cast<uint32_t>(max_size);
  if (vkCreateDescriptorPool(logicalDevice_, &poolInfo, nullptr,
                             &descriptorPool_) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create descriptor pool!");
  }
}

void VulkanController::_createDescriptorSet() {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool_;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout_;
  if (vkAllocateDescriptorSets(logicalDevice_, &allocInfo, &descriptorSet_) !=
      VK_SUCCESS) {
    throw VulkanControllerException("Failed to allocate descriptor set!");
  }
}

void VulkanController::_createFence() {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(logicalDevice_, &fenceInfo, nullptr, &computeFence_);
}

void VulkanController::_createBuffers(size_t max_size) {
  // Create CurrentLayer buffer
  _createBuffer(sizeof(GLSLNeuron) * max_size, currentLayerBufferInfo_,
                currentLayerBuffer_, currentLayerBufferMemory_);
  // Create CurrentLayerValues buffer
  _createBuffer(sizeof(cv::Vec4f) * max_size, currentLayerValuesBufferInfo_,
                currentLayerValuesBuffer_, currentLayerValuesBufferMemory_);
  // Create CurrentLayerErrors buffer
  _createBuffer(sizeof(cv::Vec4f) * max_size, currentLayerErrorsBufferInfo_,
                currentLayerErrorsBuffer_, currentLayerErrorsBufferMemory_);
  // Create currentNeighborsIndexes buffer
  _createBuffer(sizeof(uint) * max_size * MAX_NEIGHBOORS_PER_NEURON,
                currentNeighborsIndexesBufferInfo_,
                currentNeighborsIndexesBuffer_,
                currentNeighborsIndexesBufferMemory_);
  // Create currentNeighborsWeights buffer
  _createBuffer(sizeof(cv::Vec4f) * max_size * MAX_NEIGHBOORS_PER_NEURON,
                currentNeighborsWeightsBufferInfo_,
                currentNeighborsWeightsBuffer_,
                currentNeighborsWeightsBufferMemory_);
  // Create adjacentLayer buffer (i.e. PreviousLayer in forward, NextLayer in
  // backward)
  _createBuffer(sizeof(GLSLNeuron) * max_size, adjacentLayerBufferInfo_,
                adjacentLayerBuffer_, adjacentLayerBufferMemory_);
  // Create adjacentLayerValues buffer (i.e. PreviousLayerValues in forward,
  // NextLayerErrors in backward)
  _createBuffer(sizeof(cv::Vec4f) * max_size, adjacentLayerValuesBufferInfo_,
                adjacentLayerValuesBuffer_, adjacentLayerValuesBufferMemory_);
  // Create Weights buffer (i.e. CurrentLayerWeights in forward,
  // NextLayerWeights in backward)
  const auto &max_weights = Manager::getConstInstance().network->max_weights;
  _createBuffer(sizeof(cv::Vec4f) * max_size * max_weights, weightsBufferInfo_,
                weightsBuffer_, weightsBufferMemory_);
  // Create Parameters buffer
  _createBuffer(sizeof(GLSLParameters), parametersBufferInfo_,
                parametersBuffer_, parametersBufferMemory_);
  // Create Ouput buffer (i.e. CurrentLayerNewValues in forward,
  // CurrentLayerNewErrors in backward)
  _createBuffer(sizeof(cv::Vec4f) * max_size, outputBufferInfo_, outputBuffer_,
                outputBufferMemory_);
}

void VulkanController::_createBuffer(VkDeviceSize size,
                                     VkBufferCreateInfo &bufferInfo,
                                     VkBuffer buffer,
                                     VkDeviceMemory &bufferMemory) {
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size; // Size of the buffer in bytes
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // Buffer will be used
                                                         // as a storage buffer
  bufferInfo.sharingMode =
      VK_SHARING_MODE_EXCLUSIVE; // Buffer will be used by one queue family at a
                                 // time
  if (vkCreateBuffer(logicalDevice_, &bufferInfo, nullptr, &buffer) !=
      VK_SUCCESS) {
    throw VulkanControllerException("Failed to create buffer!");
  }
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(logicalDevice_, buffer, &memRequirements);
  // Allocate memory for the buffer
  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = _findMemoryType(
      memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (vkAllocateMemory(logicalDevice_, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw VulkanControllerException("Failed to allocate buffer memory!");
  }
  vkBindBufferMemory(logicalDevice_, buffer, bufferMemory, 0);
}

void VulkanController::_createDataMapping() {
  // Binding 0
  vkMapMemory(logicalDevice_, currentLayerBufferMemory_, 0,
              currentLayerBufferInfo_.size, 0, &currentLayerData_);
  // Binding 1
  vkMapMemory(logicalDevice_, currentLayerValuesBufferMemory_, 0,
              currentLayerValuesBufferInfo_.size, 0, &currentLayerValuesData_);
  // Binding 2
  vkMapMemory(logicalDevice_, currentLayerErrorsBufferMemory_, 0,
              currentLayerErrorsBufferInfo_.size, 0, &currentLayerErrorsData_);
  // Binding 3
  vkMapMemory(logicalDevice_, currentNeighborsIndexesBufferMemory_, 0,
              currentNeighborsIndexesBufferInfo_.size, 0,
              &currentNeighborsIndexesData_);
  // Binding 4
  vkMapMemory(logicalDevice_, currentNeighborsWeightsBufferMemory_, 0,
              currentNeighborsWeightsBufferInfo_.size, 0,
              &currentNeighborsWeightsData_);
  // Binding 5
  vkMapMemory(logicalDevice_, adjacentLayerBufferMemory_, 0,
              adjacentLayerBufferInfo_.size, 0, &adjacentLayerData_);
  // Binding 6
  vkMapMemory(logicalDevice_, adjacentLayerValuesBufferMemory_, 0,
              adjacentLayerValuesBufferInfo_.size, 0,
              &adjacentLayerValuesData_);
  // Binding 7
  vkMapMemory(logicalDevice_, weightsBufferMemory_, 0, weightsBufferInfo_.size,
              0, &weightsData_);
  // Binding 8
  vkMapMemory(logicalDevice_, parametersBufferMemory_, 0,
              parametersBufferInfo_.size, 0, &parametersData_);
  // Binding 9
  vkMapMemory(logicalDevice_, outputBufferMemory_, 0, outputBufferInfo_.size, 0,
              &outputData_);
}

void VulkanController::_copyNeuronsToBuffer(const NeuronMat &neurons,
                                            VkBufferCreateInfo &bufferInfo,
                                            void *&bufferData) {
  size_t totalNeuronsSize = 0;
  for (const auto &row : neurons) {
    totalNeuronsSize += row.size();
  }

  std::vector<GLSLNeuron> flatNeurons;
  flatNeurons.reserve(totalNeuronsSize);
  for (const auto &row : neurons) {
    for (const auto &neuron : row) {
      flatNeurons.push_back({.index_x = (uint)neuron.index_x,
                             .index_y = (uint)neuron.index_y,
                             .weightsIndex = (uint)neuron.weightsIndex,
                             .neighborsIndex = (uint)neuron.neighborsIndex,
                             .neighborsSize = (uint)neuron.neighborsSize});
    }
  }
  memset(bufferData, 0, (size_t)bufferInfo.size);
  memcpy(bufferData, flatNeurons.data(),
         (size_t)flatNeurons.size() * sizeof(GLSLNeuron));
}

void VulkanController::_copyMatToBuffer(const cv::Mat &mat,
                                        VkBufferCreateInfo &bufferInfo,
                                        void *&bufferData) {
  std::vector<cv::Vec4f> flatValues;
  flatValues.reserve(mat.total());
  for (size_t x = 0; x < (size_t)mat.cols; x++) {
    for (size_t y = 0; y < (size_t)mat.rows; y++) {
      flatValues.push_back(mat.at<cv::Vec4f>(x, y));
    }
  }
  memset(bufferData, 0, (size_t)bufferInfo.size);
  memcpy(bufferData, flatValues.data(),
         (size_t)flatValues.size() * sizeof(cv::Vec4f));
}

void VulkanController::_copyParametersToParametersBuffer(Layer *currentLayer) {
  const auto &network_params = Manager::getConstInstance().network_params;
  GLSLParameters parameters{
      .error_min = network_params.error_min,
      .error_max = network_params.error_max,
      .activationAlpha = currentLayer->activationFunctionAlpha,
      .currentLayerSizeX = (uint)currentLayer->size_x,
      .currentLayerSizeY = (uint)currentLayer->size_y,
      .previousLayerSizeX = currentLayer->previousLayer
                                ? (uint)currentLayer->previousLayer->size_x
                                : 0,
      .previousLayerSizeY = currentLayer->previousLayer
                                ? (uint)currentLayer->previousLayer->size_y
                                : 0,
      .nextLayerSizeX =
          currentLayer->nextLayer ? (uint)currentLayer->nextLayer->size_x : 0,
      .nextLayerSizeY =
          currentLayer->nextLayer ? (uint)currentLayer->nextLayer->size_y : 0,
      .activationFunction = (uint)currentLayer->eactivationFunction};
  memset(parametersData_, 0, (size_t)parametersBufferInfo_.size);
  memcpy(parametersData_, &parameters, sizeof(GLSLParameters));
}

// Flatten the all the weights vectors
void VulkanController::_copyNeuronsWeightsToWeightsBuffer(
    const NeuronMat &neurons) {
  // Get total sum of weights of every neurons
  size_t totalWeightsSize = 0;
  for (const auto &row : neurons) {
    totalWeightsSize += std::accumulate(row.begin(), row.end(), 0ull,
                                        [](size_t sum, const Neuron &neuron) {
                                          return sum + neuron.weights.total();
                                        });
  }
  // flatten every weights mat to a flatWeights vector
  std::vector<cv::Vec4f> flatWeights;
  flatWeights.reserve(totalWeightsSize);
  size_t weightsIndex = 0;
  for (const auto &row : neurons) {
    for (auto &neuron : row) {
      neuron.weightsIndex = weightsIndex;
      neuron.weights.forEach<cv::Vec4f>(
          [&flatWeights](const auto &value, const int *pos) {
            flatWeights.push_back(value);
          });
      weightsIndex += neuron.weights.total();
    }
  }
  // copy the flatWeights to the buffer
  memset(weightsData_, 0, (size_t)weightsBufferInfo_.size);
  memcpy(weightsData_, flatWeights.data(),
         flatWeights.size() * sizeof(cv::Vec4f));
}

// Copy the OutputBuffer data directly into the value field of the neurons
void VulkanController::_copyOutputBufferToMat(cv::Mat &mat) {
  // retrieve the data
  const auto bufferDataArray = static_cast<std::array<float, 4> *>(outputData_);

  // copy the data
  size_t index = 0;
  for (size_t x = 0; x < (size_t)mat.cols; x++) {
    for (size_t y = 0; y < (size_t)mat.rows; y++) {
      mat.at<cv::Vec4f>(x, y) =
          cv::Vec4f(bufferDataArray[index][0], bufferDataArray[index][1],
                    bufferDataArray[index][2], bufferDataArray[index][3]);
      index++;
    }
  }
  // some debug logs
  const auto &app_params = Manager::getConstInstance().app_params;
  if (app_params.verbose_debug) {
    std::atomic<bool> areAllZero = true;
    mat.forEach<cv::Vec4f>([&areAllZero](const auto &value, const int *pos) {
      if (value != cv::Vec4f::all(0.0)) {
        areAllZero = false;
      }
    });
    if (areAllZero) {
      SimpleLogger::LOG_DEBUG("Warning: all matrix values are zero.");
    }
  }
}

void VulkanController::_createPipelineLayout() {
  VkDescriptorSetLayout setLayouts[] = {descriptorSetLayout_};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;            // Optional
  pipelineLayoutInfo.pSetLayouts = setLayouts;      // Optional
  pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
  pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

  if (vkCreatePipelineLayout(logicalDevice_, &pipelineLayoutInfo, nullptr,
                             &pipelineLayout_) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create pipeline layout!");
  }
}

uint32_t
VulkanController::_findMemoryType(uint32_t typeFilter,
                                  VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw VulkanControllerException("failed to find suitable memory type!");
}

std::optional<unsigned int> VulkanController::_pickQueueFamily() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount,
                                           nullptr);
  if (queueFamilyCount == 0) {
    throw VulkanControllerException(
        "failed to find GPUs with Vulkan queue support!");
  }
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount,
                                           queueFamilies.data());
  unsigned int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
    i++;
  }
  return std::nullopt;
}

void VulkanController::_createShaderModules() {
  // forward shader
  VkShaderModuleCreateInfo createForwardInfo{};
  createForwardInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createForwardInfo.codeSize = forwardShader_->size() * sizeof(uint32_t);
  createForwardInfo.pCode = forwardShader_->data();
  if (vkCreateShaderModule(logicalDevice_, &createForwardInfo, nullptr,
                           &forwardShaderModule_) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create forward shader module");
  }
  // backward shader
  VkShaderModuleCreateInfo createBackwardInfo{};
  createBackwardInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createBackwardInfo.codeSize = backwardShader_->size() * sizeof(uint32_t);
  createBackwardInfo.pCode = backwardShader_->data();
  if (vkCreateShaderModule(logicalDevice_, &createBackwardInfo, nullptr,
                           &backwardShaderModule_) != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create backward shader module");
  }
}

void VulkanController::_createShadersComputePipelines() {
  // forward shader
  forwardPipelineInfo_.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  forwardPipelineInfo_.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  forwardPipelineInfo_.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  forwardPipelineInfo_.stage.module = forwardShaderModule_;
  forwardPipelineInfo_.stage.pName = "main";
  forwardPipelineInfo_.layout = pipelineLayout_;
  forwardPipelineInfo_.basePipelineHandle = VK_NULL_HANDLE;
  forwardPipelineInfo_.basePipelineIndex = 0;
  if (vkCreateComputePipelines(logicalDevice_, VK_NULL_HANDLE, 1,
                               &forwardPipelineInfo_, nullptr,
                               &forwardComputePipeline_) != VK_SUCCESS) {
    throw VulkanControllerException(
        "Failed to create forward compute pipelines");
  };
  // backward shader
  backwardPipelineInfo_.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  backwardPipelineInfo_.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  backwardPipelineInfo_.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  backwardPipelineInfo_.stage.module = backwardShaderModule_;
  backwardPipelineInfo_.stage.pName = "main";
  backwardPipelineInfo_.layout = pipelineLayout_;
  backwardPipelineInfo_.basePipelineHandle = VK_NULL_HANDLE;
  backwardPipelineInfo_.basePipelineIndex = 0;
  if (vkCreateComputePipelines(logicalDevice_, VK_NULL_HANDLE, 1,
                               &backwardPipelineInfo_, nullptr,
                               &backwardComputePipeline_) != VK_SUCCESS) {
    throw VulkanControllerException(
        "Failed to create backward compute pipelines");
  };
}

void VulkanController::_bindBuffers() {
  // TODO: refactor it using a container
  //  Binding 0
  VkDescriptorBufferInfo descriptor0{.buffer = currentLayerBuffer_,
                                     .offset = 0,
                                     .range = currentLayerBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet0{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor0};

  // Binding 1
  VkDescriptorBufferInfo descriptor1{.buffer = currentLayerValuesBuffer_,
                                     .offset = 0,
                                     .range =
                                         currentLayerValuesBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet1{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor1};

  // Binding 2
  VkDescriptorBufferInfo descriptor2{.buffer = currentLayerErrorsBuffer_,
                                     .offset = 0,
                                     .range =
                                         currentLayerErrorsBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet2{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor2};

  // Binding 3
  VkDescriptorBufferInfo descriptor3{
      .buffer = currentNeighborsIndexesBuffer_,
      .offset = 0,
      .range = currentNeighborsIndexesBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet3{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor3};

  // Binding 4
  VkDescriptorBufferInfo descriptor4{
      .buffer = currentNeighborsWeightsBuffer_,
      .offset = 0,
      .range = currentNeighborsWeightsBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet4{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 4,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor4};

  // Binding 5
  VkDescriptorBufferInfo descriptor5{.buffer = adjacentLayerBuffer_,
                                     .offset = 0,
                                     .range = adjacentLayerBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet5{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 5,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor5};

  // Binding 6
  VkDescriptorBufferInfo descriptor6{.buffer = adjacentLayerValuesBuffer_,
                                     .offset = 0,
                                     .range =
                                         adjacentLayerValuesBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet6{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 6,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor6};

  // Binding 7
  VkDescriptorBufferInfo descriptor7{
      .buffer = weightsBuffer_, .offset = 0, .range = weightsBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet7{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 7,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor7};

  // Binding 8
  VkDescriptorBufferInfo descriptor8{.buffer = parametersBuffer_,
                                     .offset = 0,
                                     .range = parametersBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet8{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 8,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor8};

  // Binding 9
  VkDescriptorBufferInfo descriptor9{
      .buffer = outputBuffer_, .offset = 0, .range = outputBufferInfo_.size};
  VkWriteDescriptorSet writeDescriptorSet9{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 9,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor9};

  // Update the descriptor set
  std::array<VkWriteDescriptorSet, BUFFER_COUNT> writeDescriptorSets = {
      writeDescriptorSet0, writeDescriptorSet1, writeDescriptorSet2,
      writeDescriptorSet3, writeDescriptorSet4, writeDescriptorSet5,
      writeDescriptorSet6, writeDescriptorSet7, writeDescriptorSet8,
      writeDescriptorSet9};
  vkUpdateDescriptorSets(logicalDevice_,
                         static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);
}

std::optional<VkPhysicalDevice> VulkanController::_pickPhysicalDevice() {
  auto getDeviceSuitableScore = [](const VkPhysicalDevice &device) {
    int score = 0;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    if (deviceFeatures.logicOp) {
      score++;
    }
    if (deviceFeatures.shaderFloat64) {
      score++;
    }

    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
      score += 3;
    }
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
      score += 2;
    }
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
      score += 1;
    }
    return score;
  };

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, devices.data());

  std::vector<std::pair<VkPhysicalDevice, int>> scores;
  for (const auto &device : devices) {
    scores.emplace_back(device, getDeviceSuitableScore(device));
  }
  auto betterDevice = std::max_element(
      scores.begin(), scores.end(),
      [](auto &score1, auto &score2) { return score1.second < score2.second; });
  if (betterDevice == scores.end() || betterDevice->second == 0) {
    return std::nullopt; // No suitable GPU found
  }
  return betterDevice->first;
}

void VulkanController::destroy() {
  auto freeBuffer = [this](VkBuffer &buffer, VkDeviceMemory &memory) {
    if (buffer != VK_NULL_HANDLE) {
      vkUnmapMemory(logicalDevice_, memory);
      vkFreeMemory(logicalDevice_, memory, nullptr);
      vkDestroyBuffer(logicalDevice_, buffer, nullptr);
      memory = VK_NULL_HANDLE;
      buffer = VK_NULL_HANDLE;
    }
  };

  if (forwardShaderModule_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(logicalDevice_, forwardShaderModule_, nullptr);
    forwardShaderModule_ = VK_NULL_HANDLE;
  }
  if (forwardComputePipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(logicalDevice_, forwardComputePipeline_, nullptr);
    forwardComputePipeline_ = VK_NULL_HANDLE;
  }
  if (backwardShaderModule_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(logicalDevice_, backwardShaderModule_, nullptr);
    backwardShaderModule_ = VK_NULL_HANDLE;
  }
  if (backwardComputePipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(logicalDevice_, backwardComputePipeline_, nullptr);
    backwardComputePipeline_ = VK_NULL_HANDLE;
  }

  freeBuffer(currentLayerBuffer_, currentLayerBufferMemory_);
  freeBuffer(currentLayerValuesBuffer_, currentLayerValuesBufferMemory_);
  freeBuffer(currentLayerErrorsBuffer_, currentLayerErrorsBufferMemory_);
  freeBuffer(currentNeighborsIndexesBuffer_,
             currentNeighborsIndexesBufferMemory_);
  freeBuffer(currentNeighborsWeightsBuffer_,
             currentNeighborsWeightsBufferMemory_);
  freeBuffer(adjacentLayerBuffer_, adjacentLayerBufferMemory_);
  freeBuffer(adjacentLayerValuesBuffer_, adjacentLayerValuesBufferMemory_);
  freeBuffer(weightsBuffer_, weightsBufferMemory_);
  freeBuffer(parametersBuffer_, parametersBufferMemory_);
  freeBuffer(outputBuffer_, outputBufferMemory_);

  for (auto &commmandBuffer : commandBufferPool_) {
    vkFreeCommandBuffers(logicalDevice_, commandPool_, 1, &commmandBuffer);
  }
  if (computeFence_ != VK_NULL_HANDLE) {
    vkDestroyFence(logicalDevice_, computeFence_, nullptr);
    computeFence_ = VK_NULL_HANDLE;
  }
  if (descriptorPool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(logicalDevice_, descriptorPool_, nullptr);
    descriptorPool_ = VK_NULL_HANDLE;
    // descriptor set is destroyed with the descriptor pool
    descriptorSet_ = VK_NULL_HANDLE;
  }
  if (pipelineLayout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(logicalDevice_, pipelineLayout_, nullptr);
    pipelineLayout_ = VK_NULL_HANDLE;
  }
  if (descriptorSetLayout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(logicalDevice_, descriptorSetLayout_, nullptr);
    descriptorSetLayout_ = VK_NULL_HANDLE;
  }
  if (commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(logicalDevice_, commandPool_, nullptr);
    commandPool_ = VK_NULL_HANDLE;
  }
  if (logicalDevice_ != VK_NULL_HANDLE) {
    vkDestroyDevice(logicalDevice_, nullptr);
    logicalDevice_ = VK_NULL_HANDLE;
    // queue is destroyed with the logical device
    queue_ = VK_NULL_HANDLE;
  }
  if (vkInstance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(vkInstance_, nullptr);
    vkInstance_ = VK_NULL_HANDLE;
    // physical device is destroyed with the instance
    physicalDevice_ = VK_NULL_HANDLE;
  }
  isInitialized_ = false;
}
