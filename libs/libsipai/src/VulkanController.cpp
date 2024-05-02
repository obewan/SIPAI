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

constexpr size_t COMMAND_POOL_SIZE = 1;
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

  
  _createCommandPool();
  _createCommandBufferPool();
  _createBuffers();
  _createDescriptorSetLayout();
  _createDescriptorPool();
  _createDescriptorSet();
  _createPipelineLayout();
  _createFence();
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
  _copyNeuronsWeightsToWeightsBuffer(
      currentLayer->neurons); // before others for weights index
  _copyNeuronsToBuffer(previousLayer->neurons, getBuffer(EBuffer::AdjacentLayerNeurons));
  _copyMatToBuffer(previousLayer->values, getBuffer(EBuffer::AdjacentLayerValues));
  _copyNeuronsToBuffer(currentLayer->neurons, getBuffer(EBuffer::CurrentLayerNeurons));
  _copyParametersToParametersBuffer(currentLayer);

  // Run the shader
  auto commandBuffer = _beginSingleTimeCommands();
  _computeShader(currentLayer->neurons, commandBuffer, forwardComputePipeline_);
  _endSingleTimeCommands(commandBuffer);

  // Get the results
  _copyOutputBufferToMat(currentLayer->values);
}

void VulkanController::backwardPropagation(Layer *nextLayer,
                                           Layer *currentLayer) {
  if (!IsInitialized()) {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }

  // Prepare data for the shader
  _copyNeuronsWeightsToWeightsBuffer(nextLayer->neurons); // binding 6
  _copyNeuronsToBuffer(nextLayer->neurons, getBuffer(EBuffer::AdjacentLayerNeurons)); // binding 4
  _copyNeuronsToBuffer(currentLayer->neurons, getBuffer(EBuffer::CurrentLayerNeurons)); // binding 0
  _copyMatToBuffer(currentLayer->values, getBuffer(EBuffer::CurrentLayerValues));  // binding 1
  _copyNeuronNeighboorsConnectionToBuffer(currentLayer); // binding 2 and 3
  _copyMatToBuffer(nextLayer->errors, getBuffer(EBuffer::AdjacentLayerValues));      // binding 5
  _copyParametersToParametersBuffer(currentLayer); // binding 7

  // Run the shader
  auto commandBuffer = _beginSingleTimeCommands();
  _computeShader(currentLayer->neurons, commandBuffer,
                 backwardComputePipeline_);
  _endSingleTimeCommands(commandBuffer);

  // Get the results
  _copyOutputBufferToMat(currentLayer->errors);
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
                                      VkCommandBuffer &commandBuffer,
                                      VkPipeline &pipeline) {
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
  uint32_t rows = static_cast<uint32_t>(neurons.size());
  uint32_t cols = rows > 0 ? static_cast<uint32_t>(neurons[0].size()) : 0;
  vkCmdDispatch(commandBuffer, cols, rows, 1);
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

void VulkanController::_endSingleTimeCommands(VkCommandBuffer &commandBuffer) {
  // Ends recording the command
  auto result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan command buffer end error.");
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

  // Reset the fence
  result = vkResetFences(logicalDevice_, 1, &computeFence_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan reset fence error.");
  }

  // Reset the command buffer
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
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
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
  // Buffer layout binding
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings{};
  for (size_t i = 0; i < buffers_.size(); i++) {
      VkDescriptorSetLayoutBinding layoutBinding;
      layoutBinding.binding = buffers_.at(i).binding; 
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layoutBinding.descriptorCount = 1; 
      layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; 
      layoutBindings.push_back(layoutBinding);
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(buffers_.size()); // number of bindings in the descriptor set
  layoutInfo.pBindings = layoutBindings.data(); // array of bindings
  auto result = vkCreateDescriptorSetLayout(logicalDevice_, &layoutInfo,
                                            nullptr, &descriptorSetLayout_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create descriptor set layout!");
  }
}

void VulkanController::_createDescriptorPool() {
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(buffers_.size());
  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = static_cast<uint32_t>(buffers_.size());
  auto result = vkCreateDescriptorPool(logicalDevice_, &poolInfo, nullptr,
                                       &descriptorPool_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create descriptor pool!");
  }
}

void VulkanController::_createDescriptorSet() {
  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool_;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout_;
  auto result =
      vkAllocateDescriptorSets(logicalDevice_, &allocInfo, &descriptorSet_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Failed to allocate descriptor set!");
  }
}

void VulkanController::_createFence() {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  auto result =
      vkCreateFence(logicalDevice_, &fenceInfo, nullptr, &computeFence_);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Failed to create fence!");
  }
}

void VulkanController::_createBuffers() {    
    const auto& max_size = Manager::getConstInstance().network->max_weights;
    // Initialize the vector
    for (auto [ebuffer, bufferName]:buffer_map) {        
        VkDeviceSize size = 0;
        switch (ebuffer) {
        case EBuffer::CurrentLayerNeurons: 
        case EBuffer::AdjacentLayerNeurons:
            size = sizeof(GLSLNeuron) * max_size;
            break;
        case EBuffer::CurrentLayerValues: 
        case EBuffer::AdjacentLayerValues:
        case EBuffer::Output:
            size = sizeof(cv::Vec4f) * max_size;
            break;
        case EBuffer::CurrentNeighborsErrors:
        case EBuffer::CurrentNeighborsWeights: 
            size = sizeof(cv::Vec4f) * max_size * MAX_NEIGHBOORS_PER_NEURON;
            break;
        case EBuffer::LayerWeights:
            size = sizeof(cv::Vec4f) * max_size * max_size;
            break;
        case EBuffer::Parameters:
            size = sizeof(GLSLParameters);
            break;
        default:
            throw VulkanControllerException("Buffer not implemented.");
        }
        Buffer buffer = { .name = ebuffer, .binding = (uint)ebuffer };
        buffer.info.size = size;
        buffer.info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer.info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // Storage buffers
        buffer.info.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // one queue family at a time
        if (vkCreateBuffer(logicalDevice_, &buffer.info, nullptr, &buffer.buffer) !=
            VK_SUCCESS) {
            throw VulkanControllerException("Failed to create buffer!");
        }
        // Allocate memory for the buffer
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(logicalDevice_, buffer.buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = _findMemoryType(
            memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(logicalDevice_, &allocInfo, nullptr, &buffer.memory) !=
            VK_SUCCESS) {
            throw VulkanControllerException("Failed to allocate buffer memory!");
        }
        if (vkBindBufferMemory(logicalDevice_, buffer.buffer, buffer.memory, 0) != VK_SUCCESS) {
            throw VulkanControllerException("Failed to bind buffer memory!");
        }
        buffers_.push_back(buffer);
    };    
}

void VulkanController::_createDataMapping() {
    for (auto& buffer : buffers_) {
        if (vkMapMemory(logicalDevice_, buffer.memory, 0,
            buffer.info.size, 0,
            &buffer.data) != VK_SUCCESS) {
            throw VulkanControllerException(
                "Failed to create allocate memory for " + buffer_map.at(buffer.name));
        }
        // Validation
        VkMappedMemoryRange memoryRange{};
        memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        memoryRange.memory = buffer.memory; // The device memory object
        memoryRange.offset = 0;           // Starting offset within the memory object
        memoryRange.size = VK_WHOLE_SIZE; // Size of the memory range to invalidate
        VkResult result =
            vkInvalidateMappedMemoryRanges(logicalDevice_, 1, &memoryRange);
        if (result != VK_SUCCESS) {
            throw VulkanControllerException(
                "Failed to validate memory for " + buffer_map.at(buffer.name));
       }
   }
}

void VulkanController::_copyNeuronsToBuffer(const NeuronMat &neurons, Buffer& buffer) {
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
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatNeurons.data(),
         (size_t)flatNeurons.size() * sizeof(GLSLNeuron));
}

void VulkanController::_copyMatToBuffer(const cv::Mat &mat, Buffer& buffer) {
  std::vector<cv::Vec4f> flatValues;
  flatValues.reserve(mat.total());
  for (int x = 0; x < mat.cols; x++) {
    for (int y = 0; y < mat.rows; y++) {
      flatValues.push_back(mat.at<cv::Vec4f>(x, y));
    }
  }
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatValues.data(),
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
  auto &buffer = getBuffer(EBuffer::Parameters);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &parameters, sizeof(GLSLParameters));
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
  auto& buffer = getBuffer(EBuffer::LayerWeights);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatWeights.data(),
         flatWeights.size() * sizeof(cv::Vec4f));
}
void VulkanController::_copyNeuronNeighboorsConnectionToBuffer(Layer *layer) {
  // get the neighboors connections weights and erros
  std::vector<cv::Vec4f> neighboorsConnectionWeights;
  std::vector<cv::Vec4f> neighboorsConnectionErrors;
  size_t connectionWeightsIndex = 0;
  for (const auto &row : layer->neurons) {
    for (auto &neuron : row) {
      neuron.neighborsSize = neuron.neighbors.size();
      neuron.neighborsIndex = connectionWeightsIndex;
      for (auto &connection : neuron.neighbors) {
        neighboorsConnectionWeights.push_back(connection.weight);
        neighboorsConnectionErrors.push_back(layer->errors.at<cv::Vec4f>(
            (int)connection.neuron->index_x, (int)connection.neuron->index_y));
      }
      connectionWeightsIndex += neuron.neighborsSize;
    }
  }
  // copy the weights to the buffer
  auto& bufferWeights = getBuffer(EBuffer::CurrentNeighborsWeights);
  memset(bufferWeights.data, 0, (size_t)bufferWeights.info.size);
  memcpy(bufferWeights.data, neighboorsConnectionWeights.data(),
         neighboorsConnectionWeights.size() * sizeof(cv::Vec4f));
  // copy the errors to the buffer
  auto& bufferErrors = getBuffer(EBuffer::CurrentNeighborsErrors);
  memset(bufferErrors.data, 0,
         (size_t)bufferErrors.info.size);
  memcpy(bufferErrors.data, neighboorsConnectionErrors.data(),
         neighboorsConnectionErrors.size() * sizeof(cv::Vec4f));
}

// Copy the OutputBuffer data directly into the value field of the neurons
void VulkanController::_copyOutputBufferToMat(cv::Mat &mat) {
  auto& bufferOutput = getBuffer(EBuffer::Output);
  // retrieve the data
  const auto bufferDataArray = static_cast<std::array<float, 4> *>(bufferOutput.data);

  // copy the data
  size_t index = 0;
  for (int x = 0; x < mat.cols; x++) {
    for (int y = 0; y < mat.rows; y++) {
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
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  for (auto& buffer : buffers_) {
        VkDescriptorBufferInfo descriptor{ .buffer = buffer.buffer,
                                     .offset = 0,
                                     .range = buffer.info.size };
        VkWriteDescriptorSet writeDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptorSet_,
            .dstBinding = buffer.binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &descriptor };
        writeDescriptorSets.push_back(writeDescriptorSet);
    }  
 
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
  auto freeBuffer = [this](Buffer buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
      vkUnmapMemory(logicalDevice_, buffer.memory);
      vkFreeMemory(logicalDevice_, buffer.memory, nullptr);
      vkDestroyBuffer(logicalDevice_, buffer.buffer, nullptr);
      buffer.memory = VK_NULL_HANDLE;
      buffer.buffer = VK_NULL_HANDLE;
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

  for (auto& buffer : buffers_) {
      freeBuffer(buffer);
  }

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
