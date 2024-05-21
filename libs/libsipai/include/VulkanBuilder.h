/**
 * @file VulkanBuilder.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan builder
 * @date 2024-05-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "VulkanCommon.h"
#include <optional>

namespace sipai {
class VulkanBuilder {
public:
  /**
   * @brief set the command pool size
   *
   * @param size
   * @return VulkanBuilder&
   */
  VulkanBuilder &withCommandPoolSize(size_t size = 1) {
    commandPoolSize_ = size;
    return *this;
  }
  /**
   * @brief set the max neighboors per neurons
   *
   * @param size
   * @return VulkanBuilder&
   */
  VulkanBuilder &withMaxNeighboorsPerNeuron(size_t size = 4) {
    maxNeighboosPerNeuron_ = size;
    return *this;
  }
  /**
   * @brief enable debug infos (do not enable for production)
   *
   * @param enable
   * @return VulkanBuilder&
   */
  VulkanBuilder &withDebugInfo(bool enable = true) {
    enableDebugInfo_ = enable;
    return *this;
  }

  /**
   * @brief set Vulkan struct
   *
   * @param vulkan
   * @return VulkanBuilder&
   */
  VulkanBuilder &withVulkan(std::shared_ptr<Vulkan> vulkan) {
    vulkan_ = vulkan;
    return *this;
  }

  /**
   * @brief build the Vulkan controller
   *
   * @return VulkanBuilder&
   */
  VulkanBuilder &build();

  /**
   * @brief Maps a buffer's VRAM memory to CPU RAM memory.
   * This allows the CPU to directly read from or write to the VRAM.
   * Note: Proper synchronization must be ensured to prevent data races.
   * Make sure any GPU operations using the buffer have completed before
   * mapping.
   *
   * @param buffer The buffer whose memory is to be mapped.
   */
  void mapBufferMemory(Buffer &buffer);

  /**
   * @brief Unmaps a previously mapped buffer's VRAM memory from the CPU RAM
   * memory. After this operation, the CPU will no longer have direct access to
   * the buffer's memory. Note: Any writes to the memory should be completed
   * before unmapping to ensure they are reflected in the buffer.
   *
   * @param buffer The buffer whose memory is to be unmapped.
   */
  void unmapBufferMemory(Buffer &buffer);

  /**
   * @brief initialize the vulkan instance
   *
   * @return VulkanBuilder&
   */
  VulkanBuilder &initialize();

  /**
   * @brief clear the vulkan instance
   *
   * @return VulkanBuilder&
   */
  VulkanBuilder &clear();

  /**
   * @brief find memory type
   *
   * @param typeFilter
   * @param properties
   * @return uint32_t
   */
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) const;

  /**
   * @brief Get the Memory Properties flags
   *
   * @return VkMemoryPropertyFlags
   */
  VkMemoryPropertyFlags getMemoryProperties();

  /**
   * @brief load a GLSL shader file and compile it into a uint32 vector
   *
   * @param path
   * @return std::unique_ptr<std::vector<uint32_t>>
   */
  std::unique_ptr<std::vector<uint32_t>> loadShader(const std::string &path);

private:
  std::optional<unsigned int> _pickQueueFamily();
  std::optional<VkPhysicalDevice> _pickPhysicalDevice();

  void _createCommandPool();
  void _createCommandBufferPool();
  void _createBuffers();
  void _createDescriptorSetLayout();
  void _createDescriptorPool();
  void _createDescriptorSet();
  void _createPipelineLayout();
  void _createFence();
  void _createShaderModules();
  void _createShadersComputePipelines();
  void _bindBuffers();

  std::shared_ptr<Vulkan> vulkan_ = nullptr;
  size_t commandPoolSize_ = 1;
  size_t maxNeighboosPerNeuron_ = 4;
  bool enableDebugInfo_ = false;
};
} // namespace sipai