#include "VulkanHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/VulkanHelperException.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

using namespace sipai;

bool VulkanHelper::replaceTemplateParameters(const std::string &inputFile,
                                             const std::string &outputFile) {
  std::filesystem::path pi(inputFile);
  if (!std::filesystem::exists(pi.parent_path())) {
    SimpleLogger::LOG_ERROR(
        "The input shader template directory does not exist: ",
        pi.parent_path().string());
    return false;
  }
  std::ifstream inFile(inputFile);
  if (!inFile.is_open()) {
    SimpleLogger::LOG_ERROR("Failed to open input file: ", inputFile);
    return false;
  }

  std::filesystem::path po(outputFile);
  if (!std::filesystem::exists(po.parent_path())) {
    SimpleLogger::LOG_ERROR(
        "The output shader template directory does not exist: ",
        po.parent_path().string());
    return false;
  }
  std::ofstream outFile(outputFile);
  if (!outFile.is_open()) {
    SimpleLogger::LOG_ERROR("Failed to open output file: ", outputFile);
    return false;
  }

  const auto &network_param = Manager::getConstInstance().network_params;
  size_t maxSizeX =
      std::max({network_param.input_size_x, network_param.hidden_size_x,
                network_param.output_size_x});
  size_t maxSizeY =
      std::max({network_param.input_size_y, network_param.hidden_size_y,
                network_param.output_size_y});

  std::map<std::string, std::string> values({
      {"%%MAX_SIZE_X%%", std::to_string(maxSizeX)},
      {"%%MAX_SIZE_Y%%", std::to_string(maxSizeY)},
      {"%%INPUT_SIZE_X%%", std::to_string(network_param.input_size_x)},
      {"%%INPUT_SIZE_Y%%", std::to_string(network_param.input_size_y)},
      {"%%HIDDEN_SIZE_X%%", std::to_string(network_param.hidden_size_x)},
      {"%%HIDDEN_SIZE_Y%%", std::to_string(network_param.hidden_size_y)},
      {"%%OUTPUT_SIZE_X%%", std::to_string(network_param.output_size_x)},
      {"%%OUTPUT_SIZE_Y%%", std::to_string(network_param.output_size_y)},
      {"%%OUTPUT_SIZE_XY%%", std::to_string(network_param.output_size_x *
                                            network_param.output_size_y)},
  });

  std::string line;
  size_t pos;
  while (std::getline(inFile, line)) {
    for (auto &[key, value] : values) {
      while ((pos = line.find(key)) != std::string::npos) {
        line.replace(pos, key.length(), value);
      }
    }
    outFile << line << '\n';
  }

  if (inFile.bad() || outFile.bad()) {
    SimpleLogger::LOG_ERROR("Error during GLSL templating.");
    return false;
  }

  return true;
}

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
    throw VulkanHelperException("Vulkan fence reset error.");
  }

  // Reset the command buffer
  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    throw VulkanHelperException("Vulkan command buffer reset error.");
  }

  // Ensure the device has finished all operations
  // Commented: job already done by fence
  // result = vkDeviceWaitIdle(vulkan_->logicalDevice);
  // if (result != VK_SUCCESS) {
  //   throw VulkanHelperException("Vulkan wait idle error.");
  // }

  vulkan_->commandBufferPool.push_back(commandBuffer);
}