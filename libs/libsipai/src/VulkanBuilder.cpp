#include "VulkanBuilder.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/VulkanBuilderException.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/highgui/highgui_c.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN // Reduce windows.h includes
#define NOMINMAX            // Prevent windows.h from defining min and max macros
#include <windows.h>
#endif

#ifdef _WIN32
#define VK_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan_win32.h>
#else
#include <X11/Xlib.h>
#define VK_PROTOTYPES
#define VK_USE_PLATFORM_XLIB_KHR
#include <vulkan/vulkan_xlib.h>
#endif

using namespace sipai;

VulkanBuilder &VulkanBuilder::build()
{
  const auto &app_params = Manager::getConstInstance().app_params;

  // initialize
  initialize();
  if (!vulkan_->isInitialized)
  {
    throw VulkanBuilderException("Vulkan initialization failure.");
  }

  // add vertices
  vulkan_->vertices = vertices;

  // load shaders
  for (auto &shader : vulkan_->shaders)
  {
    shader.shader = loadShader(shader.filename);
  }

  // create buffers, pipelines and others.
  _createSyncObjects();
  _createBuffers();
  _createDescriptorPool();
  _createDescriptorSetLayout();
  _allocateDescriptorSets();
  _updateDescriptorSets();
  _createShaderModules();
  _createPipelineLayout();
  if (app_params.vulkan_debug)
  {
    _createSurface();
    _createSwapChain();
    _createRenderPass();
  }
  _createShaderPipelines();
  _createCommandPool();
  _allocateCommandBuffers();
  _createFence();
  if (app_params.vulkan_debug)
  {
    _createImageViews();
    _createFramebuffers();
  }

  return *this;
}

VulkanBuilder &VulkanBuilder::initialize()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  if (vulkan_->isInitialized)
  {
    return *this;
  }

  // Get Vulkan instance
  _createInstance();

  // Get a physical device
  vulkan_->physicalDevice = _pickPhysicalDevice().value_or(VK_NULL_HANDLE);
  if (vulkan_->physicalDevice == VK_NULL_HANDLE)
  {
    throw VulkanBuilderException("Failed to find a suitable GPU!");
  }

  // Create a logical device with its queues
  _createLogicalDevice();

  // Check some properties
  bool isValid = _checkDeviceProperties();

  vulkan_->isInitialized = isValid;

  return *this;
}

void VulkanBuilder::_createInstance()
{
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "SIPAI";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  // get instance extensions
  uint32_t instanceExtensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount,
                                         nullptr);
  std::vector<VkExtensionProperties> availableInstanceExtensions(
      instanceExtensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &instanceExtensionCount,
                                         availableInstanceExtensions.data());
  bool extensionSurface = false;
  bool extensionPlateformSurface = false;
  for (const auto &extension : availableInstanceExtensions)
  {
    // Generic surface extension
    if (strcmp(VK_KHR_SURFACE_EXTENSION_NAME, extension.extensionName) == 0)
    {
      extensionSurface = true;
    }
    // Windows surface extension
#ifdef _WIN32
    if (strcmp(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, extension.extensionName) ==
        0)
    {
      extensionPlateformSurface = true;
    }
#else
    // Linux surface extension
    if (strcmp(VK_KHR_XLIB_SURFACE_EXTENSION_NAME, extension.extensionName) ==
        0)
    {
      extensionPlateformSurface = true;
    }
#endif
  }
  if (!extensionSurface)
  {
    throw VulkanBuilderException("Surface extension not found.");
  }
  if (!extensionPlateformSurface)
  {
    throw VulkanBuilderException("Plateform surface extension not found.");
  }

  const auto &app_params = Manager::getConstInstance().app_params;

  std::vector<const char *> instanceExtensions;
  if (app_params.vulkan_debug)
  {
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
    instanceExtensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif
  }

  VkInstanceCreateInfo createInfoInstance{};
  createInfoInstance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfoInstance.pApplicationInfo = &appInfo;
  createInfoInstance.enabledExtensionCount =
      static_cast<uint32_t>(instanceExtensions.size());
  createInfoInstance.ppEnabledExtensionNames = instanceExtensions.data();

  std::vector<const char *> validationLayers;
  if (app_params.vulkan_debug)
  {
    validationLayers.push_back("VK_LAYER_KHRONOS_validation");
  };
  createInfoInstance.enabledLayerCount =
      static_cast<uint32_t>(validationLayers.size());
  createInfoInstance.ppEnabledLayerNames = validationLayers.data();

  // create instance
  if (vkCreateInstance(&createInfoInstance, nullptr, &vulkan_->instance) !=
      VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create instance.");
  }
}

void VulkanBuilder::_createLogicalDevice()
{
  const auto &app_params = Manager::getConstInstance().app_params;

  // Pick graphics and compute queues
  float queuePriority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  auto graphicsQueueIndexOpt = _pickQueueGraphics();
  auto computeQueueIndexOpt = _pickQueueCompute();
  if (!graphicsQueueIndexOpt.has_value() || !computeQueueIndexOpt.has_value())
  {
    throw VulkanBuilderException("Failed to find required queue families");
  }
  vulkan_->queueGraphicsIndex = graphicsQueueIndexOpt.value();
  vulkan_->queueComputeIndex = computeQueueIndexOpt.value();

  VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
  graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  graphicsQueueCreateInfo.queueFamilyIndex = vulkan_->queueGraphicsIndex;
  graphicsQueueCreateInfo.queueCount = 1;
  graphicsQueueCreateInfo.pQueuePriorities = &queuePriority;
  queueCreateInfos.push_back(graphicsQueueCreateInfo);

  // If the compute queue index is different from the graphics queue index,
  // create a separate queue
  if (vulkan_->queueComputeIndex != vulkan_->queueGraphicsIndex)
  {
    VkDeviceQueueCreateInfo computeQueueCreateInfo{};
    computeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    computeQueueCreateInfo.queueFamilyIndex = vulkan_->queueComputeIndex;
    computeQueueCreateInfo.queueCount = 1;
    computeQueueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(computeQueueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures = {};

  // Get logical device extensions
  uint32_t deviceExtensionCount = 0;
  vkEnumerateDeviceExtensionProperties(vulkan_->physicalDevice, nullptr,
                                       &deviceExtensionCount, nullptr);
  std::vector<VkExtensionProperties> availableExtensions(deviceExtensionCount);
  vkEnumerateDeviceExtensionProperties(vulkan_->physicalDevice, nullptr,
                                       &deviceExtensionCount,
                                       availableExtensions.data());
  std::vector<const char *> deviceExtensions;
  if (app_params.vulkan_debug)
  {
    bool extensionSwapChain = false;
    bool extensionNonSemanticInfo = false;
    for (const auto &extension : availableExtensions)
    {
      if (strcmp(VK_KHR_SWAPCHAIN_EXTENSION_NAME, extension.extensionName) ==
          0)
      {
        extensionSwapChain = true;
      }
      if (strcmp(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
                 extension.extensionName) == 0)
      {
        extensionNonSemanticInfo = true;
      }
    }
    if (!extensionSwapChain)
    {
      throw VulkanBuilderException("SwapChain extension not found.");
    }
    if (!extensionNonSemanticInfo)
    {
      throw VulkanBuilderException("Non-semantic info extension not found.");
    }
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  }

  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = queueCreateInfos.data();
  createInfoDevice.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfoDevice.pEnabledFeatures = &deviceFeatures;
  createInfoDevice.enabledExtensionCount =
      static_cast<uint32_t>(deviceExtensions.size());
  createInfoDevice.ppEnabledExtensionNames = deviceExtensions.data();

  if (vkCreateDevice(vulkan_->physicalDevice, &createInfoDevice, nullptr,
                     &vulkan_->logicalDevice) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create logical device!");
  }

  vkGetDeviceQueue(vulkan_->logicalDevice, vulkan_->queueGraphicsIndex, 0,
                   &vulkan_->queueGraphics);
  vkGetDeviceQueue(vulkan_->logicalDevice, vulkan_->queueComputeIndex, 0,
                   &vulkan_->queueCompute);
}

bool VulkanBuilder::_checkDeviceProperties()
{
  const auto &network_param = Manager::getConstInstance().network_params;
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(vulkan_->physicalDevice, &deviceProperties);

  // Checking maxComputeWorkGroupInvocations
  uint32_t maxComputeWorkGroupInvocations =
      deviceProperties.limits.maxComputeWorkGroupInvocations;
  uint32_t maxComputeWorkGroupCount0 = deviceProperties.limits.maxComputeWorkGroupCount[0];
  uint32_t maxComputeWorkGroupCount1 = deviceProperties.limits.maxComputeWorkGroupCount[1];
  uint32_t maxComputeWorkGroupCount2 = deviceProperties.limits.maxComputeWorkGroupCount[2];

  SimpleLogger::LOG_INFO("Device selected: ", deviceProperties.deviceName);
  SimpleLogger::LOG_INFO("Device maxComputeWorkGroupInvocations: ", maxComputeWorkGroupInvocations);
  SimpleLogger::LOG_INFO("Device maxComputeWorkGroupCount on X: ", maxComputeWorkGroupCount0);
  SimpleLogger::LOG_INFO("Device maxComputeWorkGroupCount on Y: ", maxComputeWorkGroupCount1);
  SimpleLogger::LOG_INFO("Device maxComputeWorkGroupCount on Z: ", maxComputeWorkGroupCount2);

  size_t maxSizeX =
      std::max({network_param.input_size_x, network_param.hidden_size_x,
                network_param.output_size_x});
  size_t maxSizeY =
      std::max({network_param.input_size_y, network_param.hidden_size_y,
                network_param.output_size_y});
  if (maxSizeX * maxSizeY > maxComputeWorkGroupInvocations)
  {
    SimpleLogger::LOG_ERROR(
        "Device maxComputeWorkGroupInvocations limit (",
        maxComputeWorkGroupInvocations,
        ") is lesser than the neural network invocations requirement: ", maxSizeX * maxSizeY, " (",
        maxSizeX, "*", maxSizeY, "): FAILURE.");
    return false;
  }
  return true;
}

std::optional<VkPhysicalDevice> VulkanBuilder::_pickPhysicalDevice()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  auto getDeviceSuitableScore = [](const VkPhysicalDevice &device)
  {
    int score = 0;
    VkPhysicalDeviceProperties deviceProperties;
    VkPhysicalDeviceFeatures deviceFeatures;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
    if (deviceFeatures.logicOp)
    {
      score++;
    }
    if (deviceFeatures.geometryShader)
    {
      score++;
    }
    if (deviceFeatures.shaderFloat64)
    {
      score++;
    }
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
      score += 4;
    }
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
      score += 3;
    }
    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
    {
      score += 2;
    }
    return score;
  };

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vulkan_->instance, &deviceCount, nullptr);
  if (deviceCount == 0)
  {
    throw std::runtime_error("Failed to find GPUs with Vulkan support!");
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vulkan_->instance, &deviceCount, devices.data());
  std::vector<std::pair<VkPhysicalDevice, int>> scores;
  for (const auto &device : devices)
  {
    scores.emplace_back(device, getDeviceSuitableScore(device));
  }
  auto betterDevice = std::max_element(
      scores.begin(), scores.end(),
      [](auto &score1, auto &score2)
      { return score1.second < score2.second; });
  if (betterDevice == scores.end() || betterDevice->second == 0)
  {
    return std::nullopt; // No suitable GPU found
  }
  return betterDevice->first;
}

uint32_t VulkanBuilder::findMemoryType(uint32_t typeFilter,
                                       VkMemoryPropertyFlags properties) const
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(vulkan_->physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
  {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties)
    {
      return i;
    }
  }

  throw VulkanBuilderException("Failed to find suitable memory type.");
}

VkMemoryPropertyFlags VulkanBuilder::getMemoryProperties()
{

  // Helper function to check if a memory type has the given property flag
  auto hasMemoryPropertyFlag = [](VkMemoryPropertyFlags propertyFlags,
                                  VkMemoryPropertyFlagBits flag)
  {
    return (propertyFlags & flag) == flag;
  };

  VkPhysicalDeviceMemoryProperties memoryProperties;
  vkGetPhysicalDeviceMemoryProperties(vulkan_->physicalDevice,
                                      &memoryProperties);

  std::array<bool, 3> hasMemoryPropertyFlags{
      false, false, false}; // {hasHostCached, hasHostCoherent, hasHostVisible}

  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
  {
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
      !hasMemoryPropertyFlags[2])
  {
    throw VulkanBuilderException("Not supported memory type.");
  }

  VkMemoryPropertyFlags memoryPropertiesFlags = 0;
  if (hasMemoryPropertyFlags[0])
  {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }
  if (hasMemoryPropertyFlags[1])
  {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  }
  if (hasMemoryPropertyFlags[2])
  {
    memoryPropertiesFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  }
  return memoryPropertiesFlags;
}

std::optional<unsigned int> VulkanBuilder::_pickQueueGraphics()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vulkan_->physicalDevice,
                                           &queueFamilyCount, nullptr);
  if (queueFamilyCount == 0)
  {
    throw VulkanBuilderException(
        "Failed to find GPUs with Vulkan queue support!");
  }
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vulkan_->physicalDevice, &queueFamilyCount, queueFamilies.data());
  unsigned int i = 0;
  for (const auto &queueFamily : queueFamilies)
  {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
    {
      return i;
    }
    i++;
  }
  return std::nullopt;
}

std::optional<unsigned int> VulkanBuilder::_pickQueueCompute()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vulkan_->physicalDevice,
                                           &queueFamilyCount, nullptr);
  if (queueFamilyCount == 0)
  {
    throw VulkanBuilderException(
        "Failed to find GPUs with Vulkan queue support!");
  }
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vulkan_->physicalDevice, &queueFamilyCount, queueFamilies.data());
  unsigned int i = 0;
  for (const auto &queueFamily : queueFamilies)
  {
    if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
    {
      return i;
    }
    i++;
  }
  return std::nullopt;
}

std::unique_ptr<std::vector<uint32_t>>
VulkanBuilder::loadShader(const std::string &path)
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  if (!std::filesystem::exists(path))
  {
    throw VulkanBuilderException("GLSL file does not exist: " + path);
  }
  const auto app_params = Manager::getConstInstance().app_params;

  // Use glslangValidator to compile the GLSL shader to SPIR-V
  // -gVS: debugging infos, -V: GLSL Vulkan (-D: HLSL)
  std::stringstream sst;
#ifdef _WIN32
  sst << "glslangValidator.exe " << (app_params.vulkan_debug ? "-gVS " : "")
      << "-V -o shader.spv " << path;
#else
  sst << "glslangValidator " << (app_params.vulkan_debug ? "-gVS " : "")
      << "-V -o shader.spv " << path;
#endif
  system(sst.str().c_str());

  // Load the compiled SPIR-V into a std::vector<uint32_t>
  std::ifstream file("shader.spv", std::ios::binary | std::ios::ate);
  if (!file.good())
  {
    throw VulkanBuilderException("Failed to open SPIR-V file");
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  auto compiledShaderCode =
      std::make_unique<std::vector<uint32_t>>(size / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char *>(compiledShaderCode->data()), size))
  {
    throw VulkanBuilderException("Failed to read SPIR-V file");
  }
  return compiledShaderCode;
}

void VulkanBuilder::_createBuffers()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  VkMemoryPropertyFlags memoryPropertiesFlags = getMemoryProperties();

  const auto &network_param = Manager::getConstInstance().network_params;
  for (auto [ebuffer, bufferName] : buffer_map)
  {
    uint output_neuron_weights = 0;
    uint hidden1_neuron_weights = 0;
    size_t neuronSize = 0;
    Buffer buffer = {.name = ebuffer, .binding = (uint)ebuffer};
    buffer.info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // SSBO storage buffers (default)
    buffer.info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer.info.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE; // one queue family at a time

    // Get the buffer max bytes size
    VkDeviceSize size = 0;
    switch (ebuffer)
    {
    case EBuffer::Parameters:
      size = sizeof(GLSLParameters);
      break;
    case EBuffer::InputData:
      size = sizeof(cv::Vec4f) * network_param.input_size_x *
             network_param.input_size_y; // inputValues
      size += sizeof(cv::Vec4f) * network_param.output_size_x *
              network_param.output_size_y; // targetValues
      size += sizeof(uint);                // is_validation
      break;
    case EBuffer::OutputData:
    case EBuffer::SharedOutputValues:
      size = sizeof(cv::Vec4f) * network_param.output_size_x *
             network_param.output_size_y; // values
      break;
    case EBuffer::OutputLoss:
      size = sizeof(float); // loss
      break;
    case EBuffer::SharedOutputLoss:
      size = sizeof(float) * network_param.output_size_x *
             network_param.output_size_y; // values
      break;
    case EBuffer::InputLayer:
      size = sizeof(float) + (3 * sizeof(uint)); // attributes
      break;
    case EBuffer::OutputLayer:
      output_neuron_weights =
          (uint)(sizeof(cv::Vec4f) * network_param.hidden_size_x *
                 network_param.hidden_size_y);
      neuronSize = sizeof(GLSLNeuron) + output_neuron_weights;
      size = neuronSize * network_param.output_size_x *
             network_param.output_size_y; // OutputNeuron neurons[][]
      size += sizeof(cv::Vec4f) * network_param.output_size_x *
              network_param.output_size_y;        // vec4 errors[][]
      size += sizeof(float) + (3 * sizeof(uint)); // others attributes
      break;
    case EBuffer::HiddenLayer1:
      hidden1_neuron_weights =
          (uint)(sizeof(cv::Vec4f) * network_param.input_size_x *
                 network_param.input_size_y);
      neuronSize = sizeof(GLSLNeuron) + hidden1_neuron_weights;
      size = neuronSize * network_param.hidden_size_x *
             network_param.hidden_size_y; // HiddenNeuron neurons[][]
      size += sizeof(cv::Vec4f) * network_param.hidden_size_x *
              network_param.hidden_size_y; // vec4 values[][]
      size += sizeof(cv::Vec4f) * network_param.hidden_size_x *
              network_param.hidden_size_y;        // vec4 errors[][]
      size += sizeof(float) + (3 * sizeof(uint)); // others attributes
      break;
    case EBuffer::Vertex:
      size = sizeof(Vertex) * vulkan_->vertices.size();
      buffer.info.usage =
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; // Vertex buffer here
      break;
    default:
      throw VulkanBuilderException("Buffer not implemented.");
    }

    buffer.info.size = size;

    if (vkCreateBuffer(vulkan_->logicalDevice, &buffer.info, nullptr,
                       &buffer.buffer) != VK_SUCCESS)
    {
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
                         &buffer.memory) != VK_SUCCESS)
    {
      throw VulkanBuilderException("Failed to allocate buffer memory!");
    }
    if (vkBindBufferMemory(vulkan_->logicalDevice, buffer.buffer, buffer.memory,
                           0) != VK_SUCCESS)
    {
      throw VulkanBuilderException("Failed to bind buffer memory!");
    }
    vulkan_->buffers.push_back(buffer);
  }; // end for
}

void VulkanBuilder::_createDescriptorPool()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  std::array<VkDescriptorPoolSize, 2> poolSizes = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = static_cast<uint32_t>(vulkan_->buffers.size());

  poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[1].descriptorCount = static_cast<uint32_t>(vulkan_->buffers.size());

  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(vulkan_->buffers.size());

  VkResult result = vkCreateDescriptorPool(vulkan_->logicalDevice, &poolInfo,
                                           nullptr, &vulkan_->descriptorPool);
  if (result != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create descriptor pool!");
  }
}

void VulkanBuilder::_createDescriptorSetLayout()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  // Buffer layout binding
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings = {};
  for (size_t i = 0; i < vulkan_->buffers.size(); i++)
  {
    if (vulkan_->buffers.at(i).name == EBuffer::Vertex)
    {
      continue; // no descriptor for vertex buffers
    }
    VkDescriptorSetLayoutBinding layoutBinding;
    layoutBinding.binding = vulkan_->buffers.at(i).binding;
    layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBinding.descriptorCount = 1;
    layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    layoutBindings.push_back(layoutBinding);
  }
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(vulkan_->buffers.size() - 1);
  layoutInfo.pBindings = layoutBindings.data(); // array of bindings
  auto result =
      vkCreateDescriptorSetLayout(vulkan_->logicalDevice, &layoutInfo, nullptr,
                                  &vulkan_->descriptorSetLayout);
  if (result != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create descriptor set layout!");
  }
}

void VulkanBuilder::_allocateDescriptorSets()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = vulkan_->descriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &vulkan_->descriptorSetLayout;
  auto result = vkAllocateDescriptorSets(vulkan_->logicalDevice, &allocInfo,
                                         &vulkan_->descriptorSet);
  if (result != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to allocate descriptor set!");
  }
}

void VulkanBuilder::_updateDescriptorSets()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
  for (auto &buffer : vulkan_->buffers)
  {
    if (buffer.name == EBuffer::Vertex)
    {
      continue; // no descriptor for vertex buffers
    }
    VkDescriptorBufferInfo descriptor{
        .buffer = buffer.buffer, .offset = 0, .range = buffer.info.size};
    descriptorBufferInfos.push_back(descriptor);
  }
  size_t pos = 0;
  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  for (auto &buffer : vulkan_->buffers)
  {
    if (buffer.name == EBuffer::Vertex)
    {
      continue; // no descriptor for vertex buffers
    }
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

void VulkanBuilder::_createShaderModules()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  for (auto &shader : vulkan_->shaders)
  {
    VkShaderModuleCreateInfo createForwardInfo = {};
    createForwardInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createForwardInfo.codeSize = shader.shader->size() * sizeof(uint32_t);
    createForwardInfo.pCode = shader.shader->data();
    if (vkCreateShaderModule(vulkan_->logicalDevice, &createForwardInfo,
                             nullptr, &shader.module) != VK_SUCCESS)
    {
      throw VulkanBuilderException("Failed to create shader module of " +
                                   shader.filename);
    }
  }
}

void VulkanBuilder::_createPipelineLayout()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkDescriptorSetLayout setLayouts[] = {vulkan_->descriptorSetLayout};
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = setLayouts;
  if (vkCreatePipelineLayout(vulkan_->logicalDevice, &pipelineLayoutInfo,
                             nullptr, &vulkan_->pipelineLayout) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create pipeline layout!");
  }
}

void VulkanBuilder::_createShaderPipelines()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  std::vector<VkPipelineShaderStageCreateInfo> shaderGraphicsStages;
  std::list<VkPipelineShaderStageCreateInfo> computeShaderStages;
  for (auto &shader : vulkan_->shaders)
  {
    switch (shader.shadername)
    {
    // vertex shader
    case EShader::VertexShader:
    {
      VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
      vertShaderStageInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
      vertShaderStageInfo.module = shader.module;
      vertShaderStageInfo.pName = "main";
      shaderGraphicsStages.push_back(vertShaderStageInfo);
    }
    break;
    // fragment shader
    case EShader::FragmentShader:
    {
      VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
      fragShaderStageInfo.sType =
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
      fragShaderStageInfo.module = shader.module;
      fragShaderStageInfo.pName = "main";
      shaderGraphicsStages.push_back(fragShaderStageInfo);
    }
    break;
    // others are compute shaders
    default:
    {
      computeShaderStages.push_back({.sType =
                                        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                    .module = shader.module,
                                    .pName = "main"});
    }
    break;
    }
  }

  // get vertex infos
  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();

  vertexInputInfo.vertexBindingDescriptionCount = 1;
  vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
  vertexInputInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attributeDescriptions.size());
  vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)vulkan_->swapChainExtent.width;
  viewport.height = (float)vulkan_->swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkRect2D scissor = {};
  scissor.offset = {0, 0};
  scissor.extent = vulkan_->swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports = &viewport;
  viewportState.scissorCount = 1;
  viewportState.pScissors = &scissor;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  // create graphic pipelines
  vulkan_->graphicPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  vulkan_->graphicPipelineInfo.stageCount = (uint32_t)shaderGraphicsStages.size();
  vulkan_->graphicPipelineInfo.pStages = shaderGraphicsStages.data();
  vulkan_->graphicPipelineInfo.pVertexInputState = &vertexInputInfo;
  vulkan_->graphicPipelineInfo.pInputAssemblyState = &inputAssembly;
  vulkan_->graphicPipelineInfo.pViewportState = &viewportState;
  vulkan_->graphicPipelineInfo.pRasterizationState = &rasterizer;
  vulkan_->graphicPipelineInfo.pMultisampleState = &multisampling;
  vulkan_->graphicPipelineInfo.pColorBlendState = &colorBlending;
  vulkan_->graphicPipelineInfo.layout = vulkan_->pipelineLayout;
  vulkan_->graphicPipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  vulkan_->graphicPipelineInfo.basePipelineIndex = 0;
  vulkan_->graphicPipelineInfo.renderPass = vulkan_->renderPass;
  vulkan_->graphicPipelineInfo.subpass = 0;
  if (vkCreateGraphicsPipelines(vulkan_->logicalDevice, VK_NULL_HANDLE, 1,
                                &vulkan_->graphicPipelineInfo, nullptr,
                                &vulkan_->graphicPipeline) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create graphics pipeline");
  }

  // create compute pipelines
  for(auto& computeShaderStage : computeShaderStages)
  {
    VkComputePipelineCreateInfo computePipelineInfo = {};
    computePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineInfo.stage = computeShaderStage;
    computePipelineInfo.layout = vulkan_->pipelineLayout;
    computePipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    computePipelineInfo.basePipelineIndex = 0;
    VkPipeline pipeline;
    if (vkCreateComputePipelines(vulkan_->logicalDevice, VK_NULL_HANDLE, 1,
                                &computePipelineInfo, nullptr,
                                &pipeline) != VK_SUCCESS)
    {
      throw VulkanBuilderException("Failed to create compute pipeline");
    }
    vulkan_->computePipelines.push_back(pipeline);
  }
}

void VulkanBuilder::_createCommandPool()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = vulkan_->queueComputeIndex;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  if (vkCreateCommandPool(vulkan_->logicalDevice, &poolInfo, nullptr,
                          &vulkan_->commandPool) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create command pool!");
  }
}

void VulkanBuilder::_allocateCommandBuffers()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = vulkan_->commandPool;
  allocInfo.commandBufferCount = (uint32_t)commandPoolSize_;
  vulkan_->commandBufferPool = std::vector<VkCommandBuffer>(commandPoolSize_);
  if (vkAllocateCommandBuffers(vulkan_->logicalDevice, &allocInfo,
                               vulkan_->commandBufferPool.data()) !=
      VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to allocate command buffers!");
  }
}

void VulkanBuilder::_createFence()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  auto result = vkCreateFence(vulkan_->logicalDevice, &fenceInfo, nullptr,
                              &vulkan_->computeFence);
  if (result != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create fence!");
  }
}

void VulkanBuilder::_createSurface()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
#ifdef _WIN32
  HWND hwnd = (HWND)cvGetWindowHandle(cvWindowTitle);
  HINSTANCE hInstance = (HINSTANCE)GetWindowLongPtr(hwnd, GWLP_HINSTANCE);

  // Create Win32 surface (Windows-specific)
  VkWin32SurfaceCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
  createInfo.hwnd = hwnd;
  createInfo.hinstance = hInstance;

  if (vkCreateWin32SurfaceKHR(vulkan_->instance, &createInfo, nullptr,
                              &vulkan_->surface) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create Win32 surface");
  }
#else
  Display *dpy = XOpenDisplay(nullptr);
  Window win = (Window)cvGetWindowHandle(cvWindowTitle);

  // Create Xlib surface (Linux-specific)
  VkXlibSurfaceCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
  createInfo.dpy = dpy;
  createInfo.window = win;

  if (vkCreateXlibSurfaceKHR(vulkan_->instance, &createInfo, nullptr,
                             &vulkan_->surface) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create Xlib surface");
  }
#endif
}

void VulkanBuilder::_createSwapChain()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }

  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
          vulkan_->physicalDevice, vulkan_->surface, &surfaceCapabilities) !=
      VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to get surface capabilities");
  }

  VkSwapchainCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = vulkan_->surface;
  createInfo.minImageCount = 2;
  createInfo.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  createInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
  createInfo.imageExtent = {vulkan_->window_width, vulkan_->window_height};
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  createInfo.preTransform = surfaceCapabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(vulkan_->logicalDevice, &createInfo, nullptr,
                           &vulkan_->swapChain) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create swap chain");
  }

  uint32_t imageCount;
  vkGetSwapchainImagesKHR(vulkan_->logicalDevice, vulkan_->swapChain,
                          &imageCount, nullptr);

  vulkan_->swapChainImages.resize(imageCount);
  vkGetSwapchainImagesKHR(vulkan_->logicalDevice, vulkan_->swapChain,
                          &imageCount, vulkan_->swapChainImages.data());

  vulkan_->swapChainImageFormat = VK_FORMAT_B8G8R8A8_UNORM;
  vulkan_->swapChainExtent = {vulkan_->window_width, vulkan_->window_height};
}

void VulkanBuilder::_createImageViews()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  vulkan_->swapChainImageViews.resize(vulkan_->swapChainImages.size());

  for (size_t i = 0; i < vulkan_->swapChainImages.size(); i++)
  {
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = vulkan_->swapChainImages[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = vulkan_->swapChainImageFormat;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(vulkan_->logicalDevice, &createInfo, nullptr,
                          &vulkan_->swapChainImageViews[i]) != VK_SUCCESS)
    {
      throw VulkanBuilderException("Failed to create image views");
    }
  }
}

void VulkanBuilder::_createRenderPass()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format = vulkan_->swapChainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  VkRenderPassCreateInfo renderPassInfo{};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 1;
  renderPassInfo.pAttachments = &colorAttachment;
  renderPassInfo.subpassCount = 1;
  renderPassInfo.pSubpasses = &subpass;

  if (vkCreateRenderPass(vulkan_->logicalDevice, &renderPassInfo, nullptr,
                         &vulkan_->renderPass) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create render pass");
  }
}

void VulkanBuilder::_createFramebuffers()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  vulkan_->swapChainFramebuffers.resize(vulkan_->swapChainImageViews.size());

  for (size_t i = 0; i < vulkan_->swapChainImageViews.size(); i++)
  {
    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = vulkan_->renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &vulkan_->swapChainImageViews[i];
    framebufferInfo.width = vulkan_->swapChainExtent.width;
    framebufferInfo.height = vulkan_->swapChainExtent.height;
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(vulkan_->logicalDevice, &framebufferInfo, nullptr,
                            &vulkan_->swapChainFramebuffers[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create framebuffer");
    }
  }
}

void VulkanBuilder::_createSyncObjects()
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  if (vkCreateSemaphore(vulkan_->logicalDevice, &semaphoreInfo, nullptr,
                        &vulkan_->imageAvailableSemaphore) != VK_SUCCESS ||
      vkCreateSemaphore(vulkan_->logicalDevice, &semaphoreInfo, nullptr,
                        &vulkan_->renderFinishedSemaphore) != VK_SUCCESS ||
      vkCreateFence(vulkan_->logicalDevice, &fenceInfo, nullptr,
                    &vulkan_->inFlightFence) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create synchronization objects");
  }
}

void VulkanBuilder::mapBufferMemory(Buffer &buffer)
{
  if (vulkan_ == nullptr)
  {
    throw VulkanBuilderException("Null Vulkan pointer.");
  }
  if (buffer.isMemoryMapped)
  {
    return;
  }
  if (vkMapMemory(vulkan_->logicalDevice, buffer.memory, 0, buffer.info.size, 0,
                  &buffer.data) != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to create allocate memory for " +
                                 buffer_map.at(buffer.name));
  }
  buffer.isMemoryMapped = true;
  // Validation (disable for perfs or enable for safety)
  VkMappedMemoryRange memoryRange{};
  memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memoryRange.memory = buffer.memory; // The device memory object
  memoryRange.offset = 0;             // Starting offset within the memory
  memoryRange.size = VK_WHOLE_SIZE;   // Size of the memory range
  VkResult result =
      vkInvalidateMappedMemoryRanges(vulkan_->logicalDevice, 1, &memoryRange);
  if (result != VK_SUCCESS)
  {
    throw VulkanBuilderException("Failed to validate memory for " +
                                 buffer_map.at(buffer.name));
  }
}

void VulkanBuilder::unmapBufferMemory(Buffer &buffer)
{
  if (!buffer.isMemoryMapped)
  {
    return;
  }
  vkUnmapMemory(vulkan_->logicalDevice, buffer.memory);
  buffer.isMemoryMapped = false;
}

VulkanBuilder &VulkanBuilder::clear()
{
  if (vulkan_ == nullptr)
  {
    return *this;
  }
  if (Manager::getConstInstance().app_params.vulkan_debug)
  {
    cv::destroyWindow(cvWindowTitle);
  }

  auto freeBuffer = [](std::shared_ptr<Vulkan> vulkan, Buffer buffer)
  {
    if (buffer.buffer != VK_NULL_HANDLE)
    {
      if (buffer.isMemoryMapped)
      {
        vkUnmapMemory(vulkan->logicalDevice, buffer.memory);
        buffer.isMemoryMapped = false;
      }
      vkFreeMemory(vulkan->logicalDevice, buffer.memory, nullptr);
      vkDestroyBuffer(vulkan->logicalDevice, buffer.buffer, nullptr);
      buffer.memory = VK_NULL_HANDLE;
      buffer.buffer = VK_NULL_HANDLE;
    }
  };

  for (auto &shader : vulkan_->shaders)
  {
    if (shader.module != VK_NULL_HANDLE)
    {
      vkDestroyShaderModule(vulkan_->logicalDevice, shader.module, nullptr);
      shader.module = VK_NULL_HANDLE;
    }
  }
  vulkan_->shaders.clear();

  if (vulkan_->graphicPipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(vulkan_->logicalDevice, vulkan_->graphicPipeline,
                      nullptr);
    vulkan_->graphicPipeline = VK_NULL_HANDLE;
  }

  for(auto& pipeline: vulkan_->computePipelines)
  {
    if (pipeline != VK_NULL_HANDLE)
    {
      vkDestroyPipeline(vulkan_->logicalDevice, pipeline, nullptr);
      pipeline = VK_NULL_HANDLE;
    }
  }

  for (auto framebuffer : vulkan_->swapChainFramebuffers)
  {
    vkDestroyFramebuffer(vulkan_->logicalDevice, framebuffer, nullptr);
  }

  if (vulkan_->renderPass != VK_NULL_HANDLE)
  {
    vkDestroyRenderPass(vulkan_->logicalDevice, vulkan_->renderPass, nullptr);
    vulkan_->renderPass = VK_NULL_HANDLE;
  }

  for (auto imageView : vulkan_->swapChainImageViews)
  {
    vkDestroyImageView(vulkan_->logicalDevice, imageView, nullptr);
  }

  if (vulkan_->swapChain != VK_NULL_HANDLE)
  {
    vkDestroySwapchainKHR(vulkan_->logicalDevice, vulkan_->swapChain, nullptr);
    vulkan_->swapChain = VK_NULL_HANDLE;
  }

  if (vulkan_->surface != VK_NULL_HANDLE)
  {
    vkDestroySurfaceKHR(vulkan_->instance, vulkan_->surface, nullptr);
    vulkan_->surface = VK_NULL_HANDLE;
  }

  for (auto &buffer : vulkan_->buffers)
  {
    freeBuffer(vulkan_, buffer);
  }
  vulkan_->buffers.clear();

  for (auto &commandBuffer : vulkan_->commandBufferPool)
  {
    if (commandBuffer != VK_NULL_HANDLE)
    {
      vkFreeCommandBuffers(vulkan_->logicalDevice, vulkan_->commandPool, 1,
                           &commandBuffer);
    }
    commandBuffer = VK_NULL_HANDLE;
  }

  if (vulkan_->imageAvailableSemaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(vulkan_->logicalDevice, vulkan_->imageAvailableSemaphore,
                       nullptr);
    vulkan_->imageAvailableSemaphore = VK_NULL_HANDLE;
  }
  if (vulkan_->renderFinishedSemaphore != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(vulkan_->logicalDevice, vulkan_->renderFinishedSemaphore,
                       nullptr);
    vulkan_->renderFinishedSemaphore = VK_NULL_HANDLE;
  }

  if (vulkan_->inFlightFence != VK_NULL_HANDLE)
  {
    vkDestroyFence(vulkan_->logicalDevice, vulkan_->inFlightFence, nullptr);
    vulkan_->inFlightFence = VK_NULL_HANDLE;
  }
  if (vulkan_->computeFence != VK_NULL_HANDLE)
  {
    vkDestroyFence(vulkan_->logicalDevice, vulkan_->computeFence, nullptr);
    vulkan_->computeFence = VK_NULL_HANDLE;
  }
  if (vulkan_->descriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(vulkan_->logicalDevice, vulkan_->descriptorPool,
                            nullptr);
    vulkan_->descriptorPool = VK_NULL_HANDLE;
    // descriptor set is destroyed with the descriptor pool
    vulkan_->descriptorSet = VK_NULL_HANDLE;
  }
  if (vulkan_->pipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(vulkan_->logicalDevice, vulkan_->pipelineLayout,
                            nullptr);
    vulkan_->pipelineLayout = VK_NULL_HANDLE;
  }
  if (vulkan_->descriptorSetLayout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(vulkan_->logicalDevice,
                                 vulkan_->descriptorSetLayout, nullptr);
    vulkan_->descriptorSetLayout = VK_NULL_HANDLE;
  }
  if (vulkan_->commandPool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(vulkan_->logicalDevice, vulkan_->commandPool, nullptr);
    vulkan_->commandPool = VK_NULL_HANDLE;
  }
  if (vulkan_->logicalDevice != VK_NULL_HANDLE)
  {
    vkDestroyDevice(vulkan_->logicalDevice, nullptr);
    vulkan_->logicalDevice = VK_NULL_HANDLE;
    // queue is destroyed with the logical device
    vulkan_->queueGraphics = VK_NULL_HANDLE;
    vulkan_->queueCompute = VK_NULL_HANDLE;
  }
  if (vulkan_->instance != VK_NULL_HANDLE)
  {
    vkDestroyInstance(vulkan_->instance, nullptr);
    vulkan_->instance = VK_NULL_HANDLE;
    // physical device is destroyed with the instance
    vulkan_->physicalDevice = VK_NULL_HANDLE;
  }
  vulkan_->isInitialized = false;
  return *this;
}
