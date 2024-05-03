#include "VulkanHelper.h"
#include "exception/VulkanHelperException.h"

using namespace sipai;

VkCommandBuffer VulkanHelper::beginSingleTimeCommands() {
  // Take a command buffer from the pool
  VkCommandBuffer commandBuffer = vulkan_->commandBufferPool.back();
  vulkan_->commandBufferPool.pop_back();

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  // Starts recording the command
  auto result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan command buffer start error.");
  }
  return commandBuffer;
}

void VulkanHelper::endSingleTimeCommands(VkCommandBuffer &commandBuffer) {
  // Ends recording the command
  auto result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan command buffer end error.");
  }

  // Submit the command to the queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  result = vkQueueSubmit(vulkan_->queue, 1, &submitInfo, vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan queue submit error.");
  }
  // Wait for the fence to signal that the GPU has finished
  result = vkWaitForFences(vulkan_->logicalDevice, 1, &vulkan_->computeFence,
                           VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan wait for fence error.");
  }

  // Reset the fence
  result = vkResetFences(vulkan_->logicalDevice, 1, &vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan reset fence error.");
  }

  // Reset the command buffer
  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan reset command buffer error.");
  }

  vulkan_->commandBufferPool.push_back(commandBuffer);
}