#include "VulkanBuilder.h"
#include "Manager.h"
#include "exception/VulkanBuilderException.h"
#include <filesystem>
#include <fstream>
#include <memory>

using namespace sipai;

VulkanBuilder &VulkanBuilder::build(std::shared_ptr<Vulkan> vulkan) {
  vulkan_ = vulkan;

  // initialize
  if (!vulkan_->isInitialized && !(vulkan_->isInitialized = _initialize())) {
    throw VulkanBuilderException("Vulkan initialization failure.");
  }

  // load shaders
  for (auto &shader : vulkan_->shaders) {
    shader.shader = _loadShader(shader.filename);
  }

  // create other stuff
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

  // bind buffers
  _bindBuffers();

  return *this;
}

bool VulkanBuilder::_initialize() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
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
      VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME,
      // Commented as this non_semantic extension is not on my system,
      // but it is required for GLSL GL_EXT_debug_printf and its debugPrintEXT()
      // VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
  };

  VkInstanceCreateInfo createInfoInstance{};
  createInfoInstance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfoInstance.pApplicationInfo = &appInfo;

  if (enableDebugInfo_) {
    createInfoInstance.enabledExtensionCount =
        static_cast<uint32_t>(instanceExtensions.size());
    createInfoInstance.ppEnabledExtensionNames = instanceExtensions.data();
    createInfoInstance.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfoInstance.ppEnabledLayerNames = validationLayers.data();
  }

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
  // Commented as not required
  // deviceFeatures.logicOp = VK_TRUE; // Enable logical operation feature
  // deviceFeatures.shaderFloat64 = VK_TRUE; // Enable 64-bit floats in shader
  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = &queueCreateInfo;
  createInfoDevice.queueCreateInfoCount = 1;
  createInfoDevice.pEnabledFeatures = &deviceFeatures;
  if (vkCreateDevice(vulkan_->physicalDevice, &createInfoDevice, nullptr,
                     &vulkan_->logicalDevice) != VK_SUCCESS) {
    throw VulkanBuilderException("failed to create logical device!");
  }

  // Get a queue device
  vkGetDeviceQueue(vulkan_->logicalDevice, vulkan_->queueFamilyIndex, 0,
                   &vulkan_->queue);

  return true;
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

uint32_t
VulkanBuilder::_findMemoryType(uint32_t typeFilter,
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
VulkanBuilder::_loadShader(const std::string &path) {
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

void VulkanBuilder::_createCommandPool() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = vulkan_->queueFamilyIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(vulkan_->logicalDevice, &poolInfo, nullptr,
                          &vulkan_->commandPool) != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create command pool!");
  }
}

void VulkanBuilder::_createCommandBufferPool() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = vulkan_->commandPool;
  allocInfo.commandBufferCount = commandPoolSize;
  vulkan_->commandBufferPool = std::vector<VkCommandBuffer>(commandPoolSize);
  if (vkAllocateCommandBuffers(vulkan_->logicalDevice, &allocInfo,
                               vulkan_->commandBufferPool.data()) !=
      VK_SUCCESS) {
    throw VulkanBuilderException("Failed to allocate command buffers!");
  }
}

void VulkanBuilder::_createBuffers() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  const auto &max_size = Manager::getConstInstance().network->max_weights;
  // Initialize the vector
  for (auto [ebuffer, bufferName] : buffer_map) {
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
      size = sizeof(cv::Vec4f) * max_size * maxNeighboosPerNeuron_;
      break;
    case EBuffer::LayerWeights:
      size = sizeof(cv::Vec4f) * max_size * max_size;
      break;
    case EBuffer::Parameters:
      size = sizeof(GLSLParameters);
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
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        _findMemoryType(memRequirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(vulkan_->logicalDevice, &allocInfo, nullptr,
                         &buffer.memory) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to allocate buffer memory!");
    }
    if (vkBindBufferMemory(vulkan_->logicalDevice, buffer.buffer, buffer.memory,
                           0) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to bind buffer memory!");
    }
    vulkan_->buffers.push_back(buffer);
  };
}

void VulkanBuilder::_createDescriptorSetLayout() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  // Buffer layout binding
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings{};
  for (size_t i = 0; i < vulkan_->buffers.size(); i++) {
    VkDescriptorSetLayoutBinding layoutBinding;
    layoutBinding.binding = vulkan_->buffers.at(i).binding;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings.push_back(layoutBinding);
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(
      vulkan_->buffers.size()); // number of bindings in the descriptor set
  layoutInfo.pBindings = layoutBindings.data(); // array of bindings
  auto result =
      vkCreateDescriptorSetLayout(vulkan_->logicalDevice, &layoutInfo, nullptr,
                                  &vulkan_->descriptorSetLayout);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create descriptor set layout!");
  }
}

void VulkanBuilder::_createDescriptorPool() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorPoolSize poolSize{};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = static_cast<uint32_t>(vulkan_->buffers.size());
  VkDescriptorPoolCreateInfo poolInfo{};
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

void VulkanBuilder::_createDescriptorSet() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorSetAllocateInfo allocInfo{};
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

void VulkanBuilder::_createPipelineLayout() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkDescriptorSetLayout setLayouts[] = {vulkan_->descriptorSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;            // Optional
  pipelineLayoutInfo.pSetLayouts = setLayouts;      // Optional
  pipelineLayoutInfo.pushConstantRangeCount = 0;    // Optional
  pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

  if (vkCreatePipelineLayout(vulkan_->logicalDevice, &pipelineLayoutInfo,
                             nullptr, &vulkan_->pipelineLayout) != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create pipeline layout!");
  }
}

void VulkanBuilder::_createFence() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  auto result = vkCreateFence(vulkan_->logicalDevice, &fenceInfo, nullptr,
                              &vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanBuilderException("Failed to create fence!");
  }
}

void VulkanBuilder::_createDataMapping() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  for (auto &buffer : vulkan_->buffers) {
    if (vkMapMemory(vulkan_->logicalDevice, buffer.memory, 0, buffer.info.size,
                    0, &buffer.data) != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to create allocate memory for " +
                                   buffer_map.at(buffer.name));
    }
    // Validation
    VkMappedMemoryRange memoryRange{};
    memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memoryRange.memory = buffer.memory; // The device memory object
    memoryRange.offset = 0; // Starting offset within the memory object
    memoryRange.size = VK_WHOLE_SIZE; // Size of the memory range to invalidate
    VkResult result =
        vkInvalidateMappedMemoryRanges(vulkan_->logicalDevice, 1, &memoryRange);
    if (result != VK_SUCCESS) {
      throw VulkanBuilderException("Failed to validate memory for " +
                                   buffer_map.at(buffer.name));
    }
  }
}

void VulkanBuilder::_createShaderModules() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  for (auto &shader : vulkan_->shaders) {
    VkShaderModuleCreateInfo createForwardInfo{};
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

void VulkanBuilder::_createShadersComputePipelines() {
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

void VulkanBuilder::_bindBuffers() {
  if (vulkan_ == nullptr) {
    throw VulkanBuilderException("null vulkan pointer.");
  }
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  for (auto &buffer : vulkan_->buffers) {
    VkDescriptorBufferInfo descriptor{
        .buffer = buffer.buffer, .offset = 0, .range = buffer.info.size};
    VkWriteDescriptorSet writeDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = vulkan_->descriptorSet,
        .dstBinding = buffer.binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &descriptor};
    writeDescriptorSets.push_back(writeDescriptorSet);
  }
  vkUpdateDescriptorSets(vulkan_->logicalDevice,
                         static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);
}

VulkanBuilder &VulkanBuilder::clear() {
  if (vulkan_ == nullptr) {
    return *this;
  }
  auto freeBuffer = [](std::shared_ptr<Vulkan> vulkan, Buffer buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
      vkUnmapMemory(vulkan->logicalDevice, buffer.memory);
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

  for (auto &buffer : vulkan_->buffers) {
    freeBuffer(vulkan_, buffer);
  }

  for (auto &commmandBuffer : vulkan_->commandBufferPool) {
    vkFreeCommandBuffers(vulkan_->logicalDevice, vulkan_->commandPool, 1,
                         &commmandBuffer);
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