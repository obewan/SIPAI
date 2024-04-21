/**
 * @file VulkanController.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan Controller
 * @date 2024-04-19
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace sipai {
class VulkanController {
public:
  static VulkanController &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { instance_.reset(new VulkanController); });
    return *instance_;
  }
  static const VulkanController &getConstInstance() {
    return const_cast<const VulkanController &>(getInstance());
  }
  VulkanController(VulkanController const &) = delete;
  void operator=(VulkanController const &) = delete;
  ~VulkanController() { destroy(); }

  void initialize();

  const bool IsInitialized() { return isInitialized_.load(); }

  const VkInstance &getVkInstance() { return vkInstance_; }

  const VkPhysicalDevice &getVkPhysicalDevice() { return vkPhysicalDevice_; }

  const VkDevice &getVkDevice() { return vkLogicalDevice_; }

  /**
   * @brief Reads an image from a specified path and returns it as Image
   * parts, using Vulkan
   *
   * @param imagePath
   * @param split
   * @param withPadding
   * @param resize_x
   * @param resize_y
   */
  void loadImage(const std::string &imagePath, size_t split, bool withPadding,
                 size_t resize_x = 0, size_t resize_y = 0) const;

  /**
   * @brief Load a GLSL shader
   *
   * @param path path of a GLSL file
   */
  std::unique_ptr<std::vector<uint32_t>> loadShader(const std::string &path);

  /**
   * @brief Compute a shader
   *
   * @param computeShader
   */
  void computeShader(std::unique_ptr<std::vector<uint32_t>> &computeShader);

  /**
   * @brief Create an image Buffer in memory
   *
   * @param size
   * @param usage
   * @param properties
   * @param buffer
   * @param bufferMemory
   */
  void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags properties, VkBuffer &buffer,
                    VkDeviceMemory &bufferMemory) const;

  /**
   * @brief Create a Command Pool object
   *
   */
  void createCommandPool();

  /**
   * @brief Create a Descriptor Set object
   *
   */
  void createDescriptorSetLayout();

  /**
   * @brief Create a Descriptor Pool object
   *
   */
  void createDescriptorPool();

  /**
   * @brief Create a Descriptor Set object
   *
   */
  void createDescriptorSet();

  /**
   * @brief Create a Pipeline Layout object
   *
   */
  void createPipelineLayout();

  /**
   * @brief Update a Descriptor Set
   *
   * @param buffer
   */
  void updateDescriptorSet(VkBuffer &buffer);

  /**
   * @brief Begins a single time command buffer.
   *
   * This function allocates a command buffer and starts it with the
   * VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT flag.
   *
   * @param device The logical device that the command buffer will be created
   * on.
   * @param commandPool The command pool that the command buffer will be
   * allocated from.
   * @return The allocated and started command buffer.
   */
  VkCommandBuffer beginSingleTimeCommands(VkDevice device,
                                          VkCommandPool commandPool);

  /**
   * @brief Ends a single time command buffer.
   *
   * This function ends the command buffer, submits it to the queue, and then
   * frees it once the queue has finished executing it.
   *
   * @param device The logical device that the command buffer was created on.
   * @param commandPool The command pool that the command buffer was allocated
   * from.
   * @param commandBuffer The command buffer to end and submit.
   * @param queue The queue to submit the command buffer to.
   */
  void endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                             VkCommandBuffer commandBuffer, VkQueue queue);

  /**
   * @brief Find memory type
   *
   * @param typeFilter
   * @param properties
   * @return uint32_t
   */
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) const;

  /**
   * @brief Check if a physical device has some required features
   *
   * @param device
   * @return true
   * @return false
   */
  bool isDeviceSuitable(const VkPhysicalDevice &device);

  /**
   * @brief Destroy the device instance, cleaning ressources
   *
   */
  void destroy();

private:
  VulkanController() = default;
  static std::unique_ptr<VulkanController> instance_;

  std::atomic<bool> isInitialized_ = false;

  VkInstance vkInstance_ = VK_NULL_HANDLE;
  VkPhysicalDevice vkPhysicalDevice_ = VK_NULL_HANDLE;
  VkDevice vkLogicalDevice_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;

  unsigned int queueFamilyIndex_ = 0;

  std::unique_ptr<std::vector<uint32_t>> forwardShader_;
};
} // namespace sipai