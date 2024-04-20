#include "VulkanController.h"
#include "Manager.h"
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>

using namespace sipai;

std::unique_ptr<VulkanController> VulkanController::instance_ = nullptr;

void VulkanController::initialize() {
  if (isInitialized_) {
    return;
  }

  const auto &manager = Manager::getConstInstance();

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
    if (isDeviceSuitable(device)) {
      vkPhysicalDevice_ = device;
      break;
    }
  }

  if (vkPhysicalDevice_ == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &queueFamilyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &queueFamilyCount,
                                           queueFamilies.data());

  uint32_t queueFamilyIndex = 0;
  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      queueFamilyIndex = i;
      break;
    }
    i++;
  }

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
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

  if (vkCreateDevice(vkPhysicalDevice_, &createInfoDevice, nullptr,
                     &vkLogicalDevice_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
  forwardShader_ = loadShader(manager.app_params.forwardShader);
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
  vkMapMemory(vkLogicalDevice_, stagingBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, img.data, (size_t)imageSize);
  vkUnmapMemory(vkLogicalDevice_, stagingBufferMemory);
}

std::vector<uint32_t> VulkanController::loadShader(const std::string &path) {
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
  std::vector<uint32_t> compiledShaderCode(size / sizeof(uint32_t));
  if (!file.read(reinterpret_cast<char *>(compiledShaderCode.data()), size)) {
    throw std::runtime_error("Failed to read SPIR-V file");
  }
  return compiledShaderCode;
}

void VulkanController::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    VkMemoryPropertyFlags properties,
                                    VkBuffer &buffer,
                                    VkDeviceMemory &bufferMemory) const {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(vkLogicalDevice_, &bufferInfo, nullptr, &buffer) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(vkLogicalDevice_, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(memRequirements.memoryTypeBits, properties);

  if (vkAllocateMemory(vkLogicalDevice_, &allocInfo, nullptr, &bufferMemory) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(vkLogicalDevice_, buffer, bufferMemory, 0);
}

uint32_t
VulkanController::findMemoryType(uint32_t typeFilter,
                                 VkMemoryPropertyFlags properties) const {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

bool VulkanController::isDeviceSuitable(const VkPhysicalDevice &device) {
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
  if (vkLogicalDevice_ != VK_NULL_HANDLE) {
    vkDestroyDevice(vkLogicalDevice_, nullptr);
    vkLogicalDevice_ = VK_NULL_HANDLE;
  }
  if (vkInstance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(vkInstance_, nullptr);
    vkInstance_ = VK_NULL_HANDLE;
  }
  // No need to destroy the physical device, as managed by the instance
  vkPhysicalDevice_ = VK_NULL_HANDLE;
  isInitialized_ = false;
}
