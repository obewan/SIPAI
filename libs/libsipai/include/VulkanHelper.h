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

  bool replaceTemplateParameters(const std::string &inputFile,
                                 const std::string &outputFile);

  VkCommandBuffer commandsBegin();
  void commandsEnd_SubmitQueueGraphics(VkCommandBuffer &commandBuffer,
                                       uint32_t &imageIndex);
  void commandsEnd_SubmitQueueCompute(VkCommandBuffer &commandBuffer);

private:
  std::shared_ptr<Vulkan> vulkan_;
};
} // namespace sipai