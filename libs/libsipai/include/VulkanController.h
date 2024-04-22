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

#include "Neuron.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

namespace sipai {
class VulkanController {
public:
  static VulkanController &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { controllerInstance_.reset(new VulkanController); });
    return *controllerInstance_;
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

  const VkPhysicalDevice &getVkPhysicalDevice() { return physicalDevice_; }

  const VkDevice &getVkDevice() { return logicalDevice_; }

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
   * @param neurons
   */
  void computeShader(std::unique_ptr<std::vector<uint32_t>> &computeShader,
                     std::vector<Neuron> &neurons);

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
   * @brief Update a Descriptor Set
   *
   * @param buffer
   */
  void updateDescriptorSet(VkBuffer &buffer);

  /**
   * @brief Copy Neurons data to buffer
   *
   * @param neurons
   */
  void copyNeuronsDataToBuffer(const std::vector<Neuron> &neurons);

  /**
   * @brief Copy buffer to Neurons data
   *
   * @param neurons
   */
  void copyBufferToNeuronsData(std::vector<Neuron> &neurons);

  /**
   * @brief Destroy the device instance, cleaning ressources
   *
   */
  void destroy();

  /**
   * @brief compiled forward shader
   *
   */
  std::unique_ptr<std::vector<uint32_t>> forwardShader;

private:
  VulkanController() = default;
  static std::unique_ptr<VulkanController> controllerInstance_;

  std::atomic<bool> isInitialized_ = false;
  unsigned int queueFamilyIndex_ = 0;

  VkInstance vkInstance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice logicalDevice_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  VkBuffer inputBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory inputBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo inputBufferInfo_{};
  VkBuffer outputBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory outputBufferMemory_ = VK_NULL_HANDLE;
  VkBufferCreateInfo outputBufferInfo_{};

  VkCommandBuffer _beginSingleTimeCommands(VkDevice device,
                                           VkCommandPool commandPool);
  uint32_t _findMemoryType(uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) const;

  bool _isDeviceSuitable(const VkPhysicalDevice &device);
  void _createCommandPool();
  void _createPipelineLayout();
  void _createDescriptorSet();
  void _createDescriptorSetLayout();
  void _createDescriptorPool(size_t max_size);
  void _createNeuronsBuffers(size_t max_size);
  void _createNeuronsBuffer(VkDeviceSize size, VkBufferCreateInfo &bufferInfo,
                            VkBuffer &buffer, VkDeviceMemory &bufferMemory);
  void _endSingleTimeCommands(VkDevice device, VkCommandPool commandPool,
                              VkCommandBuffer commandBuffer, VkQueue queue);
};
} // namespace sipai