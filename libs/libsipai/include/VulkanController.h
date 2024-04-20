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
  std::vector<uint32_t> loadShader(const std::string &path);

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

  std::vector<uint32_t> forwardShader_;
};
} // namespace sipai