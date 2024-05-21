#include "VulkanControllerTest.h"
#include "ActivationFunctions.h"
#include "Manager.h"
#include "Neuron.h"
#include "SimpleLogger.h"
#include "VulkanCommon.h"
#include "exception/VulkanControllerException.h"
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

using namespace sipai;

std::unique_ptr<VulkanControllerTest>
    VulkanControllerTest::controllerInstance_ = nullptr;

bool VulkanControllerTest::initialize(bool enableDebug) {
  if (vulkan_->isInitialized) {
    return true;
  }

  const auto &manager = Manager::getConstInstance();
  if (!manager.network) {
    SimpleLogger::LOG_ERROR("No neural network found.");
    return false;
  }

  if (manager.network->layers.size() != 3) {
    SimpleLogger::LOG_ERROR(
        "The current Vulkan shader is limited to exactly 3 layers : an input "
        "layer, a hidden layer and an output layer.");
    return false;
  }

  vulkan_->shaders.clear();

  vulkan_->shaders.push_back(
      {.shadername = EShader::Test1, .filename = testFile1});

  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(enableDebug)
      .withVulkan(vulkan_)
      .build();

  return vulkan_->isInitialized;
}

float VulkanControllerTest::test1() {
  _copyParameters();

  auto &shaderTest1 = getShader(EShader::Test1);
  _computeShader(shaderTest1.pipeline);

  const auto result = _getOutputData();

  return result->loss;
}

void VulkanControllerTest::_computeShader(VkPipeline &pipeline) {
  auto commandBuffer = helper_.beginSingleTimeCommands();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkan_->pipelineLayout, 0, 1,
                          &vulkan_->descriptorSet, 0, nullptr);
  vkCmdDispatch(commandBuffer, 1, 1, 1);
  helper_.endSingleTimeCommands(commandBuffer);
}

void VulkanControllerTest::_copyParameters() {
  const auto &network_params = Manager::getConstInstance().network_params;
  GLSLParameters glslParams{
      .learning_rate = network_params.learning_rate,
      .error_min = network_params.error_min,
      .error_max = network_params.error_max,
  };
  auto &buffer = getBuffer(EBuffer::Parameters);
  if (sizeof(GLSLParameters) > (size_t)buffer.info.size) {
    throw VulkanControllerException("copy buffer overflow");
  }
  builder_.mapBufferMemory(buffer);
  // memset(buffer.data, 0, (size_t)buffer.info.size);

  // Write directly to the mapped memory
  float *data = static_cast<float *>(buffer.data);
  data[0] = 0.65f; // learning_rate
  data[1] = 0.0f;  // error_min
  data[2] = 1.0f;  // error_max
  // memcpy(buffer.data, &glslParams, sizeof(GLSLParameters));

  // Flush the mapped memory range
  VkMappedMemoryRange memoryRange{};
  memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  memoryRange.memory = buffer.memory;
  memoryRange.offset = 0;
  memoryRange.size = VK_WHOLE_SIZE;
  vkFlushMappedMemoryRanges(vulkan_->logicalDevice, 1, &memoryRange);
  builder_.unmapBufferMemory(buffer);
}

void VulkanControllerTest::_copyInputLayer() {}
void VulkanControllerTest::_copyOutputLayer() {}
void VulkanControllerTest::_copyHiddenLayer1() {}
void VulkanControllerTest::_copyInputData(const cv::Mat &inputValues,
                                          const cv::Mat &targetValues,
                                          bool is_validation) {}

std::unique_ptr<GLSLOutputData> VulkanControllerTest::_getOutputData() {
  const auto &params = Manager::getConstInstance().network_params;

  // Get loss
  auto &bufferLoss = getBuffer(EBuffer::OutputLoss);
  builder_.mapBufferMemory(bufferLoss);
  float loss = *reinterpret_cast<float *>(bufferLoss.data);
  builder_.unmapBufferMemory(bufferLoss);

  // Get outputValues
  // Commented: not required here
  return std::make_unique<GLSLOutputData>(GLSLOutputData{.loss = loss});
}