#include "VulkanBuilder.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/VulkanBuilderException.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

using namespace sipai;

VulkanBuilder &VulkanBuilder::build() {
  // initialize
  initialize();
  if (!vulkan_->isInitialized) {
    throw VulkanBuilderException("Vulkan initialization failure.");
  }

  // load shaders
  for (auto &shader : vulkan_->shaders) {
    shader.shader = loadShader(shader.filename);
  }

  // create other stuff
  _createBuffers();
  _createDescriptorPool();
  _createDescriptorSetLayout();
  _allocateDescriptorSets();
  _updateDescriptorSets();
  _createShaderModules();
  _createPipelineLayout();
  _createComputePipelines();
  _createCommandPool();
  _allocateCommandBuffers();
  _createFence();

  return *this;
}

VulkanBuilder &VulkanBuilder::initialize() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  if (vulkan_->isInitialized) {
    return *this;
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "SIPAI";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  // extensions
  const std::vector<const char *> validationLayers = {
      "VK_LAYER_KHRONOS_validation"};

  const std::vector<const char *> instanceExtensions = {
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
      VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};

  VkInstanceCreateInfo createInfoInstance{};
  createInfoInstance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfoInstance.pApplicationInfo = &appInfo;
  createInfoInstance.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfoInstance.ppEnabledExtensionNames = instanceExtensions.data();

  // FOR DEBUGGING ONLY
  createInfoInstance.enabledLayerCount =
      static_cast<uint32_t>(validationLayers.size());
  createInfoInstance.ppEnabledLayerNames = validationLayers.data();

  // create instance
  if (vkCreateInstance(&createInfoInstance, nullptr, &vulkan_->vkInstance) !=
      VK_SUCCESS) {
    throw VulkanBuilderException("failed to create instance.");
  }

  // Get a device
  auto physicalDevice = _pickPhysicalDevice();
  if (!physicalDevice.has_value()) {
    throw VulkanBuilderException("failed to find a suitable GPU!");
  }
  vulkan_->physicalDevice = physicalDevice.value();

  // Get a queue family
  auto queueFamilyIndex = _pickQueueFamily();
  if (!queueFamilyIndex.has_value()) {
    throw VulkanBuilderException(
        "failed to find GPUs with Vulkan queue support!");
  }
  vulkan_->queueFamilyIndex = queueFamilyIndex.value();

  // Create a logical device
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = vulkan_->queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};

  std::vector<const char *> deviceExtensions = {};

  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = &queueCreateInfo;
  createInfoDevice.queueCreateInfoCount = 1;
  createInfoDevice.pEnabledFeatures = &deviceFeatures;
  createInfoDevice.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfoDevice.ppEnabledExtensionNames = deviceExtensions.data();
  if (vkCreateDevice(vulkan_->physicalDevice, &createInfoDevice, nullptr,
                     &vulkan_->logicalDevice) != VK_SUCCESS) {
    throw VulkanBuilderException("failed to create logical device!");
  }

  // Get a queue device
  vkGetDeviceQueue(vulkan_->logicalDevice, vulkan_->queueFamilyIndex, 0,
                   &vulkan_->queue);

  const auto &network_param = Manager::getConstInstance().network_params;
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(vulkan_->physicalDevice, &deviceProperties);

  // Checking maxComputeWorkGroupInvocations
  uint32_t maxComputeWorkGroupInvocations =
      deviceProperties.limits.maxComputeWorkGroupInvocations;
  size_t maxSizeX =
      std::max({network_param.input_size_x, network_param.hidden_size_x,
                network_param.output_size_x});
  size_t maxSizeY =
      std::max({network_param.input_size_y, network_param.hidden_size_y,
                network_param.output_size_y});
  if (maxSizeX * maxSizeY <= maxComputeWorkGroupInvocations) {
    SimpleLogger::LOG_INFO(
        "Device workgroup maximum invocations: ",
        maxComputeWorkGroupInvocations,
        ", neural network invocations requirement: ", maxSizeX * maxSizeY, " (",
        maxSizeX, "*", maxSizeY, "): OK.");
  } else {
    SimpleLogger::LOG_ERROR(
        "Device workgroup maximum invocations: ",
        maxComputeWorkGroupInvocations,
        ", neural network invocations requirement: ", maxSizeX * maxSizeY, " (",
        maxSizeX, "*", maxSizeY, "): FAILURE.");
  }

  vulkan_->isInitialized = true;
  return *this;
}

std::optional<VkPhysicalDevice> VulkanBuilder::_pickPhysicalDevice() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }

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
  vkEnumeratePhysicalDevices(vulkan_->vkInstance, &deviceCount, nullptr);
  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkan_->vkInstance, &deviceCount, devices.data());
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

uint32_t VulkanBuilder::findMemoryType(uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) const {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(vulkan_->physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw VulkanBuilderException("failed to find suitable memory type.");
}

VkMemoryPropertyFlags VulkanBuilder::getMemoryProperties() {

  // Helper function to check if a memory type has the given property flag
  auto hasMemoryPropertyFlag = [](VkMemoryPropertyFlags propertyFlags,
                                  VkMemoryPropertyFlagBits flag) {
    return (propertyFlags & flag) == flag;
  };

  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(vulkan_->physicalDevice,
                                      &memoryProperties);

  std::array<bool, 3> hasMemoryPropertyFlags{
      false, false, false}; // {hasHostCached, hasHostCoherent, hasHostVisible}

  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    const VkMemoryPropertyFlags propertyFlags =
        memoryProperties.memoryTypes[i].propertyFlags;
    hasMemoryPropertyFlags[0] |= hasMemoryPropertyFlag(
        propertyFlags, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    hasMemoryPropertyFlags[1] |= hasMemoryPropertyFlag(
        propertyFlags, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    hasMemoryPropertyFlags[2] |= hasMemoryPropertyFlag(
        propertyFlags, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  }

  if (!hasMemoryPropertyFlags[0] && !hasMemoryPropertyFlags[1] &&
      !hasMemoryPropertyFlags[2]) {
    throw VulkanBuilderException("Not supported memory type.");
  }

  VkMemoryPropertyFlags memoryPropertiesFlags = 0;
  if (hasMemoryPropertyFlags[0]) {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (hasMemoryPropertyFlags[1]) {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (hasMemoryPropertyFlags[2]) {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  return memoryPropertiesFlags;
}

std::optional<unsigned int> VulkanBuilder::_pickQueueFamily() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vulkan_->physicalDevice,
                                           &queueFamilyCount, nullptr);
  if (queueFamilyCount == 0) {
    throw VulkanBuilderException(
        "failed to find GPUs with Vulkan queue support!");
  }
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vulkan_->physicalDevice, &queueFamilyCount, queueFamilies.data());
  unsigned int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      return i;
    }
    i++;
  }
  return std::nullopt;
}

std::unique_ptr<std::vector<uint32_t>>
VulkanBuilder::loadShader(const std::string &path) {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  if (!std::filesystem::exists(path)) {
    throw VulkanBuilderException("GLSL file does not exist: " + path);
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
    throw VulkanBuilderException("Failed to open SPIR-V file");
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto compiledShaderCode =
      std::make_unique<std::vector<uint32_t>>(size / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char *>(compiledShaderCode->data()), size)) {
    throw VulkanBuilderException("Failed to read SPIR-V file");
  }
  return compiledShaderCode;
}

void VulkanBuilder::_createBuffers() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }

  VkMemoryPropertyFlags memoryPropertiesFlags = getMemoryProperties();

  const auto &network_param = Manager::getConstInstance().network_params;
  const auto &max_size = Manager::getConstInstance().network->max_weights;
  for (auto [ebuffer, bufferName] : buffer_map) {
    VkDeviceSize size = 0;
    uint output_neuron_weights = 0;
    uint hidden1_neuron_weights = 0;
    size_t neuronSize = 0;
    size_t alignedNeuronSize = 0;
    switch (ebuffer) {
    case EBuffer::Parameters:
      size = sizeof(GLSLParameters);
      break;
    case EBuffer::InputData:
      size = sizeof(cv::Vec4f) * network_param.input_size_x *
             network_param.input_size_y; // inputValues
      size = alignedSize(size, 32);      // Align
      size += sizeof(cv::Vec4f) * network_param.output_size_x *
              network_param.output_size_y; // targetValues
      size = alignedSize(size, 32);        // Align
      size += sizeof(bool);                // is_validation
      size = alignedSize(size, 32);
      break;
    case EBuffer::OutputData:
      size = sizeof(cv::Vec4f) * network_param.output_size_x *
             network_param.output_size_y; // outputValues
      size = alignedSize(size, 32);
      break;
    case EBuffer::OutputLoss:
      size = sizeof(float); // loss
      size = alignedSize(size, 32);
      break;
    case EBuffer::InputLayer:
      size = sizeof(float) + (3 * sizeof(uint)); // attributes
      size = alignedSize(size, 32);
      break;
    case EBuffer::OutputLayer:
      output_neuron_weights =
          (uint)(sizeof(cv::Vec4f) * network_param.hidden_size_x *
                 network_param.hidden_size_y);
      neuronSize = sizeof(GLSLNeuron) + output_neuron_weights;
      alignedNeuronSize = alignedSize(neuronSize, 32); // Align to 16 bytes
      size = alignedNeuronSize * network_param.output_size_x *
             network_param.output_size_y; // OutputNeuron neurons[][]

      size += sizeof(cv::Vec4f) * network_param.output_size_x *
              network_param.output_size_y; // vec4 errors[][]
      size = alignedSize(size, 32);        // Align errors[][]

      size += sizeof(float) + (3 * sizeof(uint)); // others attributes
      size = alignedSize(size, 32);
      break;
    case EBuffer::HiddenLayer1:
      hidden1_neuron_weights =
          (uint)(sizeof(cv::Vec4f) * network_param.input_size_x *
                 network_param.input_size_y);
      neuronSize = sizeof(GLSLNeuron) + hidden1_neuron_weights;
      alignedNeuronSize = alignedSize(neuronSize, 32); // Align to 16 bytes
      size = alignedNeuronSize * network_param.hidden_size_x *
             network_param.hidden_size_y; // HiddenNeuron neurons[][]

      size += sizeof(cv::Vec4f) * network_param.hidden_size_x *
              network_param.hidden_size_y; // vec4 values[][]
      size = alignedSize(size, 32);        // Align values[][]

      size += sizeof(cv::Vec4f) * network_param.hidden_size_x *
              network_param.hidden_size_y; // vec4 errors[][]
      size = alignedSize(size, 32);        // Align errors[][]

      size += sizeof(float) + (3 * sizeof(uint)); // others attributes
      size = alignedSize(size, 32);
      break;
    default:
      throw VulkanBuilderException("Buffer not implemented.");
    }
    Buffer buffer = {.name = ebuffer, .binding = (uint)ebuffer};
    buffer.info.size = size;
    buffer.info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer.info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // Storage buffers
    buffer.info.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE; // one queue family at a time
    if (vkCreateBuffer(vulkan_->logicalDevice, &buffer.info, nullptr,
                       &buffer.buffer) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to create buffer!");
    }
    // Allocate memory for the buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan_->logicalDevice, buffer.buffer,
                                  &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, memoryPropertiesFlags);
    if (vkAllocateMemory(vulkan_->logicalDevice, &allocInfo, nullptr,
                         &buffer.memory) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to allocate buffer memory!");
    }
    if (vkBindBufferMemory(vulkan_->logicalDevice, buffer.buffer, buffer.memory,
                           0) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to bind buffer memory!");
    }
    vulkan_->buffers.push_back(buffer);
  }; // end for
}

void VulkanBuilder::_createDescriptorPool() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(vulkan_->buffers.size());
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = 1;
  poolInfo.pPoolSizes = &poolSize;
  poolInfo.maxSets = static_cast<uint32_t>(vulkan_->buffers.size());
  auto result = vkCreateDescriptorPool(vulkan_->logicalDevice, &poolInfo,
                                       nullptr, &vulkan_->descriptorPool);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create descriptor pool!");
  }
}

void VulkanBuilder::_createDescriptorSetLayout() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  // Buffer layout binding
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {};
  for (size_t i = 0; i < vulkan_->buffers.size(); i++) {
    VkDescriptorSetLayoutBinding layoutBinding;
    layoutBinding.binding = vulkan_->buffers.at(i).binding;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings.push_back(layoutBinding);
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(vulkan_->buffers.size());
  layoutInfo.pBindings = layoutBindings.data(); // array of bindings
  auto result =
      vkCreateDescriptorSetLayout(vulkan_->logicalDevice, &layoutInfo, nullptr,
                                  &vulkan_->descriptorSetLayout);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create descriptor set layout!");
  }
}

void VulkanBuilder::_allocateDescriptorSets() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = vulkan_->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vulkan_->descriptorSetLayout;
  auto result = vkAllocateDescriptorSets(vulkan_->logicalDevice, &allocInfo,
                                         &vulkan_->descriptorSet);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to allocate descriptor set!");
  }
}

void VulkanBuilder::_updateDescriptorSets() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
  for (auto &buffer : vulkan_->buffers) {
    VkDescriptorBufferInfo descriptor{
        .buffer = buffer.buffer, .offset = 0, .range = buffer.info.size};
    descriptorBufferInfos.push_back(descriptor);
  }
  size_t pos = 0;
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  for (auto &buffer : vulkan_->buffers) {
    VkWriteDescriptorSet writeDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vulkan_->descriptorSet,
        .dstBinding = buffer.binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &descriptorBufferInfos.at(pos)};
    writeDescriptorSets.push_back(writeDescriptorSet);
    pos++;
  }
  vkUpdateDescriptorSets(vulkan_->logicalDevice,
                         static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);
}

void VulkanBuilder::_createShaderModules() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  for (auto &shader : vulkan_->shaders) {
    VkShaderModuleCreateInfo createForwardInfo = {};
    createForwardInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createForwardInfo.codeSize = shader.shader->size() * sizeof(uint32_t);
    createForwardInfo.pCode = shader.shader->data();
    if (vkCreateShaderModule(vulkan_->logicalDevice, &createForwardInfo,
                             nullptr, &shader.module) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to create shader module of " +
                                   shader.filename);
    }
  }
}

void VulkanBuilder::_createPipelineLayout() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorSetLayout setLayouts[] = {vulkan_->descriptorSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = setLayouts;
  if (vkCreatePipelineLayout(vulkan_->logicalDevice, &pipelineLayoutInfo,
                             nullptr, &vulkan_->pipelineLayout) != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create pipeline layout!");
  }
}

void VulkanBuilder::_createComputePipelines() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  for (auto &shader : vulkan_->shaders) {
    // forward shader
    shader.info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    shader.info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader.info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader.info.stage.module = shader.module;
    shader.info.stage.pName = "main";
    shader.info.layout = vulkan_->pipelineLayout;
    shader.info.basePipelineHandle = VK_NULL_HANDLE;
    shader.info.basePipelineIndex = 0;
    if (vkCreateComputePipelines(vulkan_->logicalDevice, VK_NULL_HANDLE, 1,
                                 &shader.info, nullptr,
                                 &shader.pipeline) != VK_SUCCESS) {
      throw VulkanBuilderException(
          "Failed to create shader compute pipelines for " + shader.filename);
    };
  }
}

void VulkanBuilder::_createCommandPool() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = vulkan_->queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(vulkan_->logicalDevice, &poolInfo, nullptr,
                          &vulkan_->commandPool) != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create command pool!");
  }
}

void VulkanBuilder::_allocateCommandBuffers() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = vulkan_->commandPool;
  allocInfo.commandBufferCount = (uint32_t)commandPoolSize_;
  vulkan_->commandBufferPool = std::vector<VkCommandBuffer>(commandPoolSize_);
  if (vkAllocateCommandBuffers(vulkan_->logicalDevice, &allocInfo,
                               vulkan_->commandBufferPool.data()) !=
      VK_SUCCESS) {
    throw VulkanBuilderException("Failed to allocate command buffers!");
  }
}

void VulkanBuilder::_createFence() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  auto result = vkCreateFence(vulkan_->logicalDevice, &fenceInfo, nullptr,
                              &vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create fence!");
  }
}

void VulkanBuilder::mapBufferMemory(Buffer &buffer) {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  if (buffer.isMemoryMapped) {
    return;
  }
  if (vkMapMemory(vulkan_->logicalDevice, buffer.memory, 0, buffer.info.size, 0,
                  &buffer.data) != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create allocate memory for " +
                                 buffer_map.at(buffer.name));
  }
  buffer.isMemoryMapped = true;
  // Validation (disable for perfs or enable for safety)
  // VkMappedMemoryRange memoryRange{};
  // memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  // memoryRange.memory = buffer.memory; // The device memory object
  // memoryRange.offset = 0;             // Starting offset within the memory
  // memoryRange.size = VK_WHOLE_SIZE;   // Size of the memory range
  // VkResult result =
  //     vkInvalidateMappedMemoryRanges(vulkan_->logicalDevice, 1,
  //     &memoryRange);
  // if (result != VK_SUCCESS) {
  //   throw VulkanBuilderException("Failed to validate memory for " +
  //                                buffer_map.at(buffer.name));
  // }
}

void VulkanBuilder::unmapBufferMemory(Buffer &buffer) {
  if (!buffer.isMemoryMapped) {
    return;
  }
  vkUnmapMemory(vulkan_->logicalDevice, buffer.memory);
  buffer.isMemoryMapped = false;
}

VulkanBuilder &VulkanBuilder::clear() {
  if (vulkan_ == nullptr) {
    return *this;
  }
  auto freeBuffer = [](std::shared_ptr<Vulkan> vulkan, Buffer buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
      if (buffer.isMemoryMapped) {
        vkUnmapMemory(vulkan->logicalDevice, buffer.memory);
        buffer.isMemoryMapped = false;
      }
      vkFreeMemory(vulkan->logicalDevice, buffer.memory, nullptr);
      vkDestroyBuffer(vulkan->logicalDevice, buffer.buffer, nullptr);
      buffer.memory = VK_NULL_HANDLE;
      buffer.buffer = VK_NULL_HANDLE;
    }
  };

  for (auto &shader : vulkan_->shaders) {
    if (shader.module != VK_NULL_HANDLE) {
      vkDestroyShaderModule(vulkan_->logicalDevice, shader.module, nullptr);
      shader.module = VK_NULL_HANDLE;
    }
    if (shader.pipeline != VK_NULL_HANDLE) {
      vkDestroyPipeline(vulkan_->logicalDevice, shader.pipeline, nullptr);
      shader.pipeline = VK_NULL_HANDLE;
    }
  }
  vulkan_->shaders.clear();

  for (auto &buffer : vulkan_->buffers) {
    freeBuffer(vulkan_, buffer);
  }
  vulkan_->buffers.clear();

  for (auto &commandBuffer : vulkan_->commandBufferPool) {
    if (commandBuffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(vulkan_->logicalDevice, vulkan_->commandPool, 1,
                           &commandBuffer);
    }
    commandBuffer = VK_NULL_HANDLE;
  }
  if (vulkan_->computeFence != VK_NULL_HANDLE) {
    vkDestroyFence(vulkan_->logicalDevice, vulkan_->computeFence, nullptr);
    vulkan_->computeFence = VK_NULL_HANDLE;
  }
  if (vulkan_->descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(vulkan_->logicalDevice, vulkan_->descriptorPool,
                            nullptr);
    vulkan_->descriptorPool = VK_NULL_HANDLE;
    // descriptor set is destroyed with the descriptor pool
    vulkan_->descriptorSet = VK_NULL_HANDLE;
  }
  if (vulkan_->pipelineLayout != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(vulkan_->logicalDevice, vulkan_->pipelineLayout,
                            nullptr);
    vulkan_->pipelineLayout = VK_NULL_HANDLE;
  }
  if (vulkan_->descriptorSetLayout != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(vulkan_->logicalDevice,
                                 vulkan_->descriptorSetLayout, nullptr);
    vulkan_->descriptorSetLayout = VK_NULL_HANDLE;
  }
  if (vulkan_->commandPool != VK_NULL_HANDLE) {
    vkDestroyCommandPool(vulkan_->logicalDevice, vulkan_->commandPool, nullptr);
    vulkan_->commandPool = VK_NULL_HANDLE;
  }
  if (vulkan_->logicalDevice != VK_NULL_HANDLE) {
    vkDestroyDevice(vulkan_->logicalDevice, nullptr);
    vulkan_->logicalDevice = VK_NULL_HANDLE;
    // queue is destroyed with the logical device
    vulkan_->queue = VK_NULL_HANDLE;
  }
  if (vulkan_->vkInstance != VK_NULL_HANDLE) {
    vkDestroyInstance(vulkan_->vkInstance, nullptr);
    vulkan_->vkInstance = VK_NULL_HANDLE;
    // physical device is destroyed with the instance
    vulkan_->physicalDevice = VK_NULL_HANDLE;
  }
  vulkan_->isInitialized = false;
  return *this;
}
