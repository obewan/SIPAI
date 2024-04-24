#include "VulkanController.h"
#include "ActivationFunctions.h"
#include "Manager.h"
#include "Neuron.h"
#include "SimpleLogger.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

using namespace sipai;

const size_t totalBuffers = 5;

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
    throw std::runtime_error("failed to create instance!");
  }

  // Create a device
  auto physicalDevice = _pickPhysicalDevice();
  if (!physicalDevice.has_value()) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }
  physicalDevice_ = physicalDevice.value();

  auto queueFamilyIndex = _pickQueueFamily();
  if (!queueFamilyIndex.has_value()) {
    throw std::runtime_error("failed to find GPUs with Vulkan queue support!");
  }
  queueFamilyIndex_ = queueFamilyIndex.value();

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex_;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.logicOp = VK_TRUE; // Enable logical operation feature
  deviceFeatures.shaderFloat64 =
      VK_TRUE; // Enable 64-bit floats in shader code feature

  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = &queueCreateInfo;
  createInfoDevice.queueCreateInfoCount = 1;
  createInfoDevice.pEnabledFeatures = &deviceFeatures;

  if (vkCreateDevice(physicalDevice_, &createInfoDevice, nullptr,
                     &logicalDevice_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
  vkGetDeviceQueue(logicalDevice_, queueFamilyIndex_, 0, &queue_);

  forwardShader = loadShader(manager.app_params.forwardShader);

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

  isInitialized_ = true;
}

void VulkanController::forwardPropagation(Layer *previousLayer,
                                          Layer *currentLayer) {
  if (!IsInitialized()) {
    throw NeuralNetworkException("Vulkan controller is not initialized.");
  }

  // Prepare data for the shader
  auto commandBuffer = commandStart();
  copyNeuronsWeightsToWeightsBuffer(
      currentLayer->neurons); // before others for weights index
  copyNeuronsDataToInputBuffer(previousLayer->neurons);
  copyActivationFunctionToActivationFunctionBuffer(
      currentLayer->activationFunction, currentLayer->activationFunctionAlpha);
  commandEnd(commandBuffer);

  commandBuffer = commandStart();
  copyNeuronsDataToCurrentBuffer(currentLayer->neurons);
  commandEnd(commandBuffer);

  // Run the shader
  computeShader(forwardShader, currentLayer->neurons);

  // Get the results
  commandBuffer = commandStart();
  copyOutputBufferToNeuronsData(currentLayer->neurons);
  commandEnd(commandBuffer);
}

std::unique_ptr<std::vector<uint32_t>>
VulkanController::loadShader(const std::string &path) {
  if (!std::filesystem::exists(path)) {
    throw std::runtime_error("GLSL file does not exist: " + path);
  }
  // Use glslangValidator to compile the GLSL shader to SPIR-V
  std::stringstream sst;
  sst << "glslangValidator -V " << path << " -o shader.spv";
  system(sst.str().c_str());

  // Load the compiled SPIR-V into a std::vector<uint32_t>
  std::ifstream file("shader.spv", std::ios::binary | std::ios::ate);
  if (!file.good()) {
    throw std::runtime_error("Failed to open SPIR-V file");
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto compiledShaderCode =
      std::make_unique<std::vector<uint32_t>>(size / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char *>(compiledShaderCode->data()), size)) {
    throw std::runtime_error("Failed to read SPIR-V file");
  }
  return compiledShaderCode;
}

void VulkanController::computeShader(
    std::unique_ptr<std::vector<uint32_t>> &computeShader,
    std::vector<Neuron> &neurons) {

  // Create shader module
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = computeShader->size() * sizeof(uint32_t);
  createInfo.pCode = computeShader->data();
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(logicalDevice_, &createInfo, nullptr,
                           &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create shader module");
  }

  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo{};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  pipelineInfo.stage.module = shaderModule;
  pipelineInfo.stage.pName = "main";
  pipelineInfo.layout = pipelineLayout_;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.basePipelineIndex = 0;
  VkPipeline computePipeline;
  if (vkCreateComputePipelines(logicalDevice_, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &computePipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create compute pipelines");
  };

  // Note: beware refactoring those following bindings
  // Bind the input buffer
  VkDescriptorBufferInfo descriptorInputBufferInfo{
      .buffer = inputBuffer_, .offset = 0, .range = inputBufferInfo_.size};
  VkWriteDescriptorSet writeInputDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 0,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorInputBufferInfo};

  // Bind the output buffer
  VkDescriptorBufferInfo descriptorOutputBufferInfo{
      .buffer = outputBuffer_, .offset = 0, .range = outputBufferInfo_.size};
  VkWriteDescriptorSet writeOutputDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 1,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorOutputBufferInfo};

  // Bind the current buffer
  VkDescriptorBufferInfo descriptorCurrentBufferInfo{
      .buffer = currentBuffer_, .offset = 0, .range = currentBufferInfo_.size};
  VkWriteDescriptorSet writeCurrentDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 2,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorCurrentBufferInfo};

  // Bind the activation function buffer
  VkDescriptorBufferInfo descriptorAFBufferInfo{
      .buffer = activationFunctionBuffer_,
      .offset = 0,
      .range = activationFunctionBufferInfo_.size};
  VkWriteDescriptorSet writeAFDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 3,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorAFBufferInfo};

  // Bind the weights buffer
  VkDescriptorBufferInfo descriptorWeightsBufferInfo{
      .buffer = weightsBuffer_, .offset = 0, .range = weightsBufferInfo_.size};
  VkWriteDescriptorSet writeWeightsDescriptorSet{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = descriptorSet_,
      .dstBinding = 4,
      .dstArrayElement = 0,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptorWeightsBufferInfo};

  // Update the descriptor set
  std::array<VkWriteDescriptorSet, totalBuffers> writeDescriptorSets = {
      writeInputDescriptorSet, writeOutputDescriptorSet,
      writeCurrentDescriptorSet, writeAFDescriptorSet,
      writeWeightsDescriptorSet};
  vkUpdateDescriptorSets(logicalDevice_,
                         static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);

  // Create command buffer and record commands
  VkCommandBuffer commandBuffer = _beginSingleTimeCommands();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    computePipeline);

  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
  vkCmdDispatch(commandBuffer, static_cast<uint32_t>(neurons.size()), 1, 1);

  _endSingleTimeCommands(commandBuffer);

  // Cleanup
  vkDestroyShaderModule(logicalDevice_, shaderModule, nullptr);
  vkDestroyPipeline(logicalDevice_, computePipeline, nullptr);
}

VkCommandBuffer VulkanController::_beginSingleTimeCommands() {
  // Take a command buffer from the pool
  VkCommandBuffer commandBuffer = commandBufferPool_.back();
  commandBufferPool_.pop_back();

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  // Starts recording the command
  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  return commandBuffer;
}

void VulkanController::_endSingleTimeCommands(VkCommandBuffer commandBuffer) {
  // Ends recording the command
  vkEndCommandBuffer(commandBuffer);
  // Reset the fence
  vkResetFences(logicalDevice_, 1, &computeFence_);
  // Submit the command to the queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(queue_, 1, &submitInfo, computeFence_);
  // Wait for the fence to signal that the GPU has finished
  vkWaitForFences(logicalDevice_, 1, &computeFence_, VK_TRUE, UINT64_MAX);

  vkResetCommandBuffer(commandBuffer, 0);
  commandBufferPool_.push_back(commandBuffer);
}

void VulkanController::_createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex =
      queueFamilyIndex_; // Index of the queue family the pool is for
  poolInfo.flags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; // Optional flags

  if (vkCreateCommandPool(logicalDevice_, &poolInfo, nullptr, &commandPool_) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to create command pool!");
  }
}

void VulkanController::_createCommandBufferPool() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool_;
  allocInfo.commandBufferCount = COMMAND_POOL_SIZE;
  std::vector<VkCommandBuffer> commandBuffers(COMMAND_POOL_SIZE);
  if (vkAllocateCommandBuffers(logicalDevice_, &allocInfo,
                               commandBuffers.data()) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate command buffers!");
  }
  commandBufferPool_ = std::move(commandBuffers);
}

void VulkanController::_createDescriptorSetLayout() {
  std::array<VkDescriptorSetLayoutBinding, totalBuffers> layoutBindings{};
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
    throw std::runtime_error("Failed to create descriptor set layout!");
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
    throw std::runtime_error("Failed to create descriptor pool!");
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
    throw std::runtime_error("Failed to allocate descriptor set!");
  }
}

void VulkanController::_createFence() {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  vkCreateFence(logicalDevice_, &fenceInfo, nullptr, &computeFence_);
}

void VulkanController::_createBuffers(size_t max_size) {
  // Create Input buffer
  _createBuffer(sizeof(Neuron) * max_size, inputBufferInfo_, inputBuffer_,
                inputBufferMemory_);
  // Create Ouput buffer
  _createBuffer(sizeof(RGBA) * max_size, outputBufferInfo_, outputBuffer_,
                outputBufferMemory_);
  // Create Current buffer
  _createBuffer(sizeof(Neuron) * max_size, currentBufferInfo_, currentBuffer_,
                currentBufferMemory_);
  // Create Activation buffer
  _createBuffer(sizeof(GLSLActivationFunction), activationFunctionBufferInfo_,
                activationFunctionBuffer_, activationFunctionBufferMemory_);
  // Create Weights buffer
  const auto &max_weights = Manager::getConstInstance().network->max_weights;
  _createBuffer(sizeof(RGBA) * max_size * max_weights, weightsBufferInfo_,
                weightsBuffer_, weightsBufferMemory_);
}

void VulkanController::_createBuffer(VkDeviceSize size,
                                     VkBufferCreateInfo &bufferInfo,
                                     VkBuffer &buffer,
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
    throw std::runtime_error("Failed to create buffer!");
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
    throw std::runtime_error("Failed to allocate buffer memory!");
  }
  vkBindBufferMemory(logicalDevice_, buffer, bufferMemory, 0);
}

void VulkanController::_createDataMapping() {
  vkMapMemory(logicalDevice_, inputBufferMemory_, 0, inputBufferInfo_.size, 0,
              &inputData_);
  vkMapMemory(logicalDevice_, currentBufferMemory_, 0, currentBufferInfo_.size,
              0, &currentData_);
  vkMapMemory(logicalDevice_, activationFunctionBufferMemory_, 0,
              activationFunctionBufferInfo_.size, 0, &activationFunctionData_);
  vkMapMemory(logicalDevice_, weightsBufferMemory_, 0, weightsBufferInfo_.size,
              0, &weightsData_);
  vkMapMemory(logicalDevice_, outputBufferMemory_, 0, outputBufferInfo_.size, 0,
              &outputData_);
}

void VulkanController::copyNeuronsDataToInputBuffer(
    const std::vector<Neuron> &neurons) {
  memset(inputData_, 0, (size_t)inputBufferInfo_.size);
  memcpy(inputData_, neurons.data(), neurons.size() * sizeof(Neuron));
}

void VulkanController::copyNeuronsDataToCurrentBuffer(
    const std::vector<Neuron> &neurons) {
  memset(currentData_, 0, (size_t)currentBufferInfo_.size);
  memcpy(currentData_, neurons.data(), neurons.size() * sizeof(Neuron));
}

void VulkanController::copyActivationFunctionToActivationFunctionBuffer(
    const EActivationFunction &activationFunction, float alpha) {
  GLSLActivationFunction glslActivationFunction{
      .value = (int)activationFunction, .alpha = alpha};
  memset(activationFunctionData_, 0,
         (size_t)activationFunctionBufferInfo_.size);
  memcpy(activationFunctionData_, &glslActivationFunction,
         sizeof(GLSLActivationFunction));
}

// Flatten the all the weights vectors
void VulkanController::copyNeuronsWeightsToWeightsBuffer(
    const std::vector<Neuron> &neurons) {
  size_t totalWeightsSize =
      std::accumulate(neurons.begin(), neurons.end(), 0ull,
                      [](size_t sum, const Neuron &neuron) {
                        return sum + neuron.weights.size();
                      });
  std::vector<RGBA> flatWeights;
  flatWeights.reserve(totalWeightsSize);
  size_t weightsIndex = 0;
  for (auto &neuron : neurons) {
    neuron.weightsIndex = weightsIndex;
    flatWeights.insert(flatWeights.end(), neuron.weights.begin(),
                       neuron.weights.end());
    weightsIndex += neuron.weights.size();
  }
  memset(weightsData_, 0, (size_t)weightsBufferInfo_.size);
  memcpy(weightsData_, flatWeights.data(), flatWeights.size() * sizeof(RGBA));
}

// Copy the OutputBuffer data directly into the value field of the neurons
void VulkanController::copyOutputBufferToNeuronsData(
    std::vector<Neuron> &neurons) {
  const auto &bufferData = static_cast<std::array<float, 4> *>(outputData_);
  for (size_t i = 0; i < neurons.size(); i++) {
    neurons[i].value.value = bufferData[i];
  }
  const auto &app_params = Manager::getConstInstance().app_params;
  if (app_params.verbose_debug &&
      std::all_of(neurons.begin(), neurons.end(), [](const auto &neuron) {
        return neuron.value.value ==
               std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
      })) {
    SimpleLogger::LOG_DEBUG("Warning: all neurons values are zero.");
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
    throw std::runtime_error("Failed to create pipeline layout!");
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

  throw std::runtime_error("failed to find suitable memory type!");
}

std::optional<unsigned int> VulkanController::_pickQueueFamily() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount,
                                           nullptr);
  if (queueFamilyCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan queue support!");
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

std::optional<VkPhysicalDevice> VulkanController::_pickPhysicalDevice() {
  auto getDeviceSuitableScore = [](const VkPhysicalDevice &device) {
    int score = 0;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    // This feature indicates whether the physical device supports logical
    // operations in the output color blending. Logical operations are used to
    // combine color and/or alpha components of the source and destination
    // fragments in ways other than traditional blending+
    if (deviceFeatures.logicOp) {
      score++;
    }
    // This feature indicates whether the physical device supports 64-bit floats
    // (doubles) in shader code. If this feature is not enabled, you won’t be
    // able to use 64-bit floats in your shaders.
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

  freeBuffer(inputBuffer_, inputBufferMemory_);
  freeBuffer(outputBuffer_, outputBufferMemory_);
  freeBuffer(currentBuffer_, currentBufferMemory_);
  freeBuffer(activationFunctionBuffer_, activationFunctionBufferMemory_);
  freeBuffer(weightsBuffer_, weightsBufferMemory_);

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
