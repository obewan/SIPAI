/**
 * @file VulkanHelper.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan Helper
 * @date 2024-05-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "VulkanCommon.h"
#include <vulkan/vulkan.hpp>

namespace sipai {
class VulkanHelper {
public:
  void setVulkan(std::shared_ptr<Vulkan> vulkan) { vulkan_ = vulkan; }

  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer &commandBuffer);

private:
  std::shared_ptr<Vulkan> vulkan_;
};
} // namespace sipai