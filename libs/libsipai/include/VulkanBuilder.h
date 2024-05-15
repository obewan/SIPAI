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
    commandPoolSize = size;
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
   * @brief build the Vulkan controller
   *
   * @param vulkan
   * @return VulkanBuilder&
   */
  VulkanBuilder &build(std::shared_ptr<Vulkan> vulkan);

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
   * @brief clear the vulkan instance
   *
   * @return VulkanBuilder&
   */
  VulkanBuilder &clear();

private:
  bool _initialize();
  uint32_t _findMemoryType(uint32_t typeFilter,
                           VkMemoryPropertyFlags properties) const;
  std::optional<unsigned int> _pickQueueFamily();
  std::optional<VkPhysicalDevice> _pickPhysicalDevice();
  std::unique_ptr<std::vector<uint32_t>> _loadShader(const std::string &path);

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
  size_t commandPoolSize = 1;
  size_t maxNeighboosPerNeuron_ = 4;
  bool enableDebugInfo_ = false;
};
} // namespace sipai