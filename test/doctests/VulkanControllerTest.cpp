#include "VulkanControllerTest.h"
#include "ActivationFunctions.h"
#include "LayerHidden.h"
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

bool VulkanControllerTest::initialize() {
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

  if (vulkan_->shaders.empty()) {
    // add more shaders there
    vulkan_->shaders.push_back(
        {.shadername = EShader::Test1, .filename = testFile1});
    vulkan_->shaders.push_back(
        {.shadername = EShader::Test2, .filename = testFile2});
  }

  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(manager.app_params.verbose_debug)
      .withVulkan(vulkan_)
      .build();

  return vulkan_->isInitialized;
}

float VulkanControllerTest::test1() {
  _copyParameters();

  auto &shaderTest1 = getShader(EShader::Test1);
  _computeShader(vulkan_->pipelineComputeTraining);

  float loss = _readOutputLoss();

  return loss;
}

VulkanControllerTest::ResultTest2 VulkanControllerTest::test2() {
  _copyParameters();
  _copyHiddenLayer1();

  auto &shaderTest2 = getShader(EShader::Test2);
  _computeShader(vulkan_->pipelineComputeTraining);

  ResultTest2 result;

  const auto loss = _readOutputLoss();
  const auto layer = _getHiddenLayer1();

  result.loss = loss;
  result.layer = layer;

  return result;
}

void VulkanControllerTest::_computeShader(VkPipeline &pipeline) {
  auto commandBuffer = helper_.commandsBegin();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  VkDescriptorSet descriptorSets[] = {vulkan_->descriptorSet};
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkan_->pipelineLayout, 0, 1, descriptorSets, 0,
                          nullptr);
  vkCmdDispatch(commandBuffer, 1, 1, 1);
  helper_.commandsEnd_SubmitQueueCompute(commandBuffer);
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
  memcpy(buffer.data, &glslParams, sizeof(GLSLParameters));

  // Flush the mapped memory range
  // VkMappedMemoryRange memoryRange{};
  // memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  // memoryRange.memory = buffer.memory;
  // memoryRange.offset = 0;
  // memoryRange.size = VK_WHOLE_SIZE;
  // vkFlushMappedMemoryRanges(vulkan_->logicalDevice, 1, &memoryRange);
  builder_.unmapBufferMemory(buffer);
}

void VulkanControllerTest::_copyInputLayer() {}
void VulkanControllerTest::_copyOutputLayer() {}
void VulkanControllerTest::_copyInputData(const cv::Mat &inputValues,
                                          const cv::Mat &targetValues,
                                          bool is_validation) {}

void VulkanControllerTest::_copyHiddenLayer1() {
  const auto &layers = Manager::getConstInstance().network->layers;
  if (layers.size() < 2) {
    throw VulkanControllerException("Invalid layers size.");
  }
  const auto &hiddenLayer1 = layers.at(1);
  if (hiddenLayer1->layerType != LayerType::LayerHidden) {
    throw VulkanControllerException("Invalid Hidden layer type.");
  }
  // Copy the layer into the VRAM
  try {
    auto &buffer = getBuffer(EBuffer::HiddenLayer1);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the neurons
    for (const auto &row : hiddenLayer1->neurons) {
      for (const auto &neuron : row) {
        // index_xy
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_x);
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_y);

        // weights
        for (int y = 0; y < neuron.weights.rows; y++) {
          for (int x = 0; x < neuron.weights.cols; x++) {
            for (int k = 0; k < 4; k++) {
              bufferPtr = copyToBuffer<float>(
                  bufferPtr, neuron.weights.at<cv::Vec4f>(y, x)[k]);
            }
          }
        }

        // neighbors
        bool isUsed = false;
        for (int i = 0; i < MAX_NEIGHBORS; i++) {
          if (i < (int)neuron.neighbors.size()) {
            isUsed = true;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr,
                                               static_cast<uint32_t>(isUsed));
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_x);
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_y);
            for (int k = 0; k < 4; k++) {
              bufferPtr =
                  copyToBuffer<float>(bufferPtr, neuron.neighbors[i].weight[k]);
            }
          } else {
            isUsed = false;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr,
                                               static_cast<uint32_t>(isUsed));
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            for (int k = 0; k < 4; k++) {
              bufferPtr = copyToBuffer<float>(bufferPtr, 0.0f);
            }
          }
        }
      }
    }

    // Copy the values
    for (int y = 0; y < hiddenLayer1->values.rows; y++) {
      for (int x = 0; x < hiddenLayer1->values.cols; x++) {
        for (int k = 0; k < 4; k++) {
          bufferPtr = copyToBuffer<float>(
              bufferPtr, hiddenLayer1->values.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the errors
    for (int y = 0; y < hiddenLayer1->errors.rows; y++) {
      for (int x = 0; x < hiddenLayer1->errors.cols; x++) {
        for (int k = 0; k < 4; k++) {
          bufferPtr = copyToBuffer<float>(
              bufferPtr, hiddenLayer1->errors.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the attributes
    bufferPtr =
        copyToBuffer<float>(bufferPtr, hiddenLayer1->activationFunctionAlpha);
    bufferPtr = copyToBuffer<uint32_t>(
        bufferPtr, (uint32_t)hiddenLayer1->eactivationFunction);
    bufferPtr =
        copyToBuffer<uint32_t>(bufferPtr, (uint32_t)hiddenLayer1->size_x);
    bufferPtr =
        copyToBuffer<uint32_t>(bufferPtr, (uint32_t)hiddenLayer1->size_y);

    builder_.unmapBufferMemory(buffer);

    size_t totalBytesCopied = bufferPtr - bufferStart;
    if (totalBytesCopied > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Hidden layer copy error: " +
                                    std::string(ex.what()));
  }
}

float VulkanControllerTest::_readOutputLoss() {
  // Get loss
  auto &bufferLoss = getBuffer(EBuffer::OutputLoss);
  builder_.mapBufferMemory(bufferLoss);
  float loss = *reinterpret_cast<float *>(bufferLoss.data);
  builder_.unmapBufferMemory(bufferLoss);

  return loss;
}

Layer *VulkanControllerTest::_getHiddenLayer1() {
  const auto &layers = Manager::getConstInstance().network->layers;
  if (layers.size() < 2) {
    throw VulkanControllerException("Invalid layers size.");
  }
  const auto &hiddenLayer1 = layers.at(1);
  if (hiddenLayer1->layerType != LayerType::LayerHidden) {
    throw VulkanControllerException("Invalid Hidden layer type.");
  }
  auto &bufferHiddenLayer = getBuffer(EBuffer::HiddenLayer1);

  builder_.mapBufferMemory(bufferHiddenLayer);
  if (!bufferHiddenLayer.data) {
    builder_.unmapBufferMemory(bufferHiddenLayer);
    throw VulkanControllerException(
        "Invalid data pointer after mapping buffer memory");
  }

  uint offset = 0;

  // Read neurons
  for (size_t y = 0; y < hiddenLayer1->size_y; ++y) {
    for (size_t x = 0; x < hiddenLayer1->size_x; ++x) {
      auto &dstNeuron = hiddenLayer1->neurons[y][x];

      // Check index_x and index_y
      uint32_t index_x =
          getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
      uint32_t index_y =
          getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
      if (dstNeuron.index_x != index_x || dstNeuron.index_y != index_y) {
        builder_.unmapBufferMemory(bufferHiddenLayer);
        throw VulkanControllerException("Invalid data buffer memory");
      }

      // Get weights
      for (int i = 0; i < dstNeuron.weights.rows; ++i) {
        for (int j = 0; j < dstNeuron.weights.cols; ++j) {
          auto value = cv::Vec4f(
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
          dstNeuron.weights.at<cv::Vec4f>(i, j) = value;
        }
      }

      // Get neighbors
      for (int i = 0; i < MAX_NEIGHBORS; i++) {
        bool isUsed = static_cast<bool>(
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset));
        uint32_t neigh_index_x =
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
        uint32_t neigh_index_y =
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);

        // Some checks
        if (isUsed &&
            (dstNeuron.neighbors[i].neuron->index_x != neigh_index_x ||
             dstNeuron.neighbors[i].neuron->index_y != neigh_index_y)) {
          builder_.unmapBufferMemory(bufferHiddenLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }
        if ((isUsed && i + 1 > (int)dstNeuron.neighbors.size()) ||
            (!isUsed && i + 1 < (int)dstNeuron.neighbors.size())) {
          builder_.unmapBufferMemory(bufferHiddenLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }

        // Get connection weight
        auto weight =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        if (isUsed) {
          dstNeuron.neighbors[i].weight = weight;
        }
      }
    } // end read neurons

    // Get values
    for (int y = 0; y < hiddenLayer1->values.rows; ++y) {
      for (int x = 0; x < hiddenLayer1->values.cols; ++x) {
        auto value =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        hiddenLayer1->values.at<cv::Vec4f>(y, x) = value;
      }
    }

    // Get errors
    for (int y = 0; y < hiddenLayer1->errors.rows; ++y) {
      for (int x = 0; x < hiddenLayer1->errors.cols; ++x) {
        auto error =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        hiddenLayer1->errors.at<cv::Vec4f>(y, x) = error;
      }
    }

    // Get others attributes (offset update)
    getDataFromBuffer<float>(bufferHiddenLayer.data,
                             offset); // activation_alpha
    getDataFromBuffer<uint>(bufferHiddenLayer.data,
                            offset); // activation_function
    getDataFromBuffer<uint>(bufferHiddenLayer.data, offset); // size_x
    getDataFromBuffer<uint>(bufferHiddenLayer.data, offset); // size_y
  }

  builder_.unmapBufferMemory(bufferHiddenLayer);
  return hiddenLayer1;
}