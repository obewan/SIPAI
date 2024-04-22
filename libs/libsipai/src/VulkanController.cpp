#include "VulkanController.h"
#include "Manager.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <vulkan/vulkan_core.h>

using namespace sipai;

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
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, devices.data());

  for (const auto &device : devices) {
    if (_isDeviceSuitable(device)) {
      physicalDevice_ = device;
      break;
    }
  }

  if (physicalDevice_ == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount,
                                           queueFamilies.data());

  ;
  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      queueFamilyIndex_ = i;
      break;
    }
    i++;
  }

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
  _createDescriptorSetLayout();
  _createDescriptorPool(max_size);
  _createDescriptorSet();
  _createPipelineLayout();
  _createNeuronsBuffers(max_size);

  isInitialized_ = true;
}

void VulkanController::loadImage(const std::string &imagePath, size_t split,
                                 bool withPadding, size_t resize_x,
                                 size_t resize_y) const {
  // Load image using OpenCV
  cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

  // Convert image data to RGBA (if not already)
  cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);

  // Create a Vulkan buffer and copy the image data into it
  VkDeviceSize imageSize = img.total() * img.elemSize();
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               stagingBuffer, stagingBufferMemory);

  // Map the buffer memory and copy the image data
  void *data;
  vkMapMemory(logicalDevice_, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, img.data, (size_t)imageSize);
  vkUnmapMemory(logicalDevice_, stagingBufferMemory);
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
  VkPipeline computePipeline;
  if (vkCreateComputePipelines(logicalDevice_, VK_NULL_HANDLE, 1, &pipelineInfo,
                               nullptr, &computePipeline) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create compute pipelines");
  };

  // Create command buffer and record commands
  VkCommandBuffer commandBuffer =
      _beginSingleTimeCommands(logicalDevice_, commandPool_);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    computePipeline);
  // Bind the input buffer
  VkDescriptorBufferInfo descriptorInputBufferInfo{};
  descriptorInputBufferInfo.buffer = inputBuffer_;
  descriptorInputBufferInfo.offset = 0;
  descriptorInputBufferInfo.range = inputBufferInfo_.size;

  VkWriteDescriptorSet writeInputDescriptorSet{};
  writeInputDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeInputDescriptorSet.dstSet = descriptorSet_;
  writeInputDescriptorSet.dstBinding = 0;
  writeInputDescriptorSet.dstArrayElement = 0;
  writeInputDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeInputDescriptorSet.descriptorCount = 1;
  writeInputDescriptorSet.pBufferInfo = &descriptorInputBufferInfo;

  // Bind the output buffer
  VkDescriptorBufferInfo descriptorOutputBufferInfo{};
  descriptorOutputBufferInfo.buffer = outputBuffer_;
  descriptorOutputBufferInfo.offset = 0;
  descriptorOutputBufferInfo.range = outputBufferInfo_.size;

  VkWriteDescriptorSet writeOutputDescriptorSet{};
  writeOutputDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writeOutputDescriptorSet.dstSet = descriptorSet_;
  writeOutputDescriptorSet.dstBinding = 1;
  writeOutputDescriptorSet.dstArrayElement = 0;
  writeOutputDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writeOutputDescriptorSet.descriptorCount = 1;
  writeOutputDescriptorSet.pBufferInfo = &descriptorOutputBufferInfo;

  // Update the descriptor set
  std::array<VkWriteDescriptorSet, 2> writeDescriptorSets = {
      writeInputDescriptorSet, writeOutputDescriptorSet};
  vkUpdateDescriptorSets(logicalDevice_,
                         static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);

  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
  vkCmdDispatch(commandBuffer, static_cast<uint32_t>(neurons.size()), 1, 1);
  _endSingleTimeCommands(logicalDevice_, commandPool_, commandBuffer, queue_);

  // Cleanup
  vkDestroyShaderModule(logicalDevice_, shaderModule, nullptr);
  vkDestroyPipeline(logicalDevice_, computePipeline, nullptr);
}

VkCommandBuffer
VulkanController::_beginSingleTimeCommands(VkDevice device,
                                           VkCommandPool commandPool) {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  return commandBuffer;
}

void VulkanController::_endSingleTimeCommands(VkDevice device,
                                              VkCommandPool commandPool,
                                              VkCommandBuffer commandBuffer,
                                              VkQueue queue) {
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);

  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanController::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags properties,
                                    VkBuffer &buffer,
                                    VkDeviceMemory &bufferMemory) const {
  // TODO: refactor, clean buffer and memory
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(logicalDevice_, &bufferInfo, nullptr, &buffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(logicalDevice_, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      _findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(logicalDevice_, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(logicalDevice_, buffer, bufferMemory, 0);
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

void VulkanController::_createDescriptorSetLayout() {
  std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};

  // Input buffer binding
  layoutBindings[0].binding = 0; // binding number
  layoutBindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // type of the bound descriptor(s)
  layoutBindings[0].descriptorCount = 1; // number of descriptors in the binding
  layoutBindings[0].stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT; // shader stages that can access the binding

  // Output buffer binding
  layoutBindings[1].binding = 1; // binding number
  layoutBindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // type of the bound descriptor(s)
  layoutBindings[1].descriptorCount = 1; // number of descriptors in the binding
  layoutBindings[1].stageFlags =
      VK_SHADER_STAGE_COMPUTE_BIT; // shader stages that can access the binding

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

void VulkanController::_createNeuronsBuffers(size_t max_size) {
  // Create Input buffer
  _createNeuronsBuffer(sizeof(Neuron) * max_size, inputBufferInfo_,
                       inputBuffer_, inputBufferMemory_);
  // Create Ouput buffer
  _createNeuronsBuffer(sizeof(RGBA) * max_size, outputBufferInfo_,
                       outputBuffer_, outputBufferMemory_);
}

void VulkanController::_createNeuronsBuffer(VkDeviceSize size,
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

void VulkanController::copyNeuronsDataToBuffer(
    const std::vector<Neuron> &neurons) {
  void *data;
  vkMapMemory(logicalDevice_, inputBufferMemory_, 0, inputBufferInfo_.size, 0,
              &data);
  memcpy(data, neurons.data(), (size_t)inputBufferInfo_.size);
  vkUnmapMemory(logicalDevice_, inputBufferMemory_);
}

void VulkanController::copyBufferToNeuronsData(std::vector<Neuron> &neurons) {
  void *data;
  vkMapMemory(logicalDevice_, outputBufferMemory_, 0, outputBufferInfo_.size, 0,
              &data);
  memcpy(static_cast<void *>(neurons.data()), data,
         (size_t)outputBufferInfo_.size);
  vkUnmapMemory(logicalDevice_, outputBufferMemory_);
}

void VulkanController::updateDescriptorSet(VkBuffer &buffer) {
  VkDescriptorBufferInfo bufferInfo{};
  bufferInfo.buffer = buffer;
  bufferInfo.offset = 0;
  bufferInfo.range = sizeof(RGBA); // TODO: CHECK THIS

  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = descriptorSet_;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &bufferInfo;

  vkUpdateDescriptorSets(logicalDevice_, 1, &descriptorWrite, 0, nullptr);
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

bool VulkanController::_isDeviceSuitable(const VkPhysicalDevice &device) {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  // This feature indicates whether the physical device supports logical
  // operations in the output color blending. Logical operations are used to
  // combine color and/or alpha components of the source and destination
  // fragments in ways other than traditional blending+
  bool logicOpShaderSupported = deviceFeatures.logicOp;

  // This feature indicates whether the physical device supports 64-bit floats
  // (doubles) in shader code. If this feature is not enabled, you won’t be
  // able to use 64-bit floats in your shaders.
  bool anisotropicFilteringSupported = deviceFeatures.shaderFloat64;

  return logicOpShaderSupported && anisotropicFilteringSupported;
}

void VulkanController::destroy() {

  if (inputBuffer_ != VK_NULL_HANDLE) {
    vkFreeMemory(logicalDevice_, inputBufferMemory_, nullptr);
    vkDestroyBuffer(logicalDevice_, inputBuffer_, nullptr);
    inputBufferMemory_ = VK_NULL_HANDLE;
    inputBuffer_ = VK_NULL_HANDLE;
  }
  if (outputBuffer_ != VK_NULL_HANDLE) {
    vkFreeMemory(logicalDevice_, outputBufferMemory_, nullptr);
    vkDestroyBuffer(logicalDevice_, outputBuffer_, nullptr);
    outputBufferMemory_ = VK_NULL_HANDLE;
    outputBuffer_ = VK_NULL_HANDLE;
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
