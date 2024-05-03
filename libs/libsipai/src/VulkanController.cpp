#include "VulkanController.h"
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
#include <opencv2/core/matx.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

using namespace sipai;

std::unique_ptr<VulkanController> VulkanController::controllerInstance_ =
    nullptr;

void VulkanController::initialize(bool enableDebug) {
  if (vulkan_->isInitialized) {
    return;
  }

  const auto &manager = Manager::getConstInstance();
  if (!manager.network) {
    return;
  }

  vulkan_->shaders.clear();
  vulkan_->shaders.push_back({.shadername = EShader::Forward,
                              .filename = manager.app_params.forwardShader});
  vulkan_->shaders.push_back({.shadername = EShader::Backward,
                              .filename = manager.app_params.backwardShader});

  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(enableDebug)
      .build(vulkan_);
}

void VulkanController::forwardPropagation(Layer *previousLayer,
                                          Layer *currentLayer) {
  if (!IsInitialized()) {
    throw NeuralNetworkException("Vulkan controller is not initialized.");
  }

  // Prepare data for the shader
  _copyNeuronsWeightsToWeightsBuffer(
      currentLayer->neurons); // before others for weights index
  _copyNeuronsToBuffer(previousLayer->neurons,
                       getBuffer(EBuffer::AdjacentLayerNeurons));
  _copyMatToBuffer(previousLayer->values,
                   getBuffer(EBuffer::AdjacentLayerValues));
  _copyNeuronsToBuffer(currentLayer->neurons,
                       getBuffer(EBuffer::CurrentLayerNeurons));
  _copyParametersToParametersBuffer(currentLayer);

  // Run the shader
  auto commandBuffer = _beginSingleTimeCommands();
  _computeShader(currentLayer->neurons, commandBuffer,
                 getShader(EShader::Forward).pipeline);
  _endSingleTimeCommands(commandBuffer);

  // Get the results
  _copyOutputBufferToMat(currentLayer->values);
}

void VulkanController::backwardPropagation(Layer *nextLayer,
                                           Layer *currentLayer) {
  if (!IsInitialized()) {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }

  // Prepare data for the shader
  _copyNeuronsWeightsToWeightsBuffer(nextLayer->neurons); // binding 6
  _copyNeuronsToBuffer(nextLayer->neurons,
                       getBuffer(EBuffer::AdjacentLayerNeurons)); // binding 4
  _copyNeuronsToBuffer(currentLayer->neurons,
                       getBuffer(EBuffer::CurrentLayerNeurons)); // binding 0
  _copyMatToBuffer(currentLayer->values,
                   getBuffer(EBuffer::CurrentLayerValues)); // binding 1
  _copyNeuronNeighboorsConnectionToBuffer(currentLayer);    // binding 2 and 3
  _copyMatToBuffer(nextLayer->errors,
                   getBuffer(EBuffer::AdjacentLayerValues)); // binding 5
  _copyParametersToParametersBuffer(currentLayer);           // binding 7

  // Run the shader
  auto commandBuffer = _beginSingleTimeCommands();
  _computeShader(currentLayer->neurons, commandBuffer,
                 getShader(EShader::Backward).pipeline);
  _endSingleTimeCommands(commandBuffer);

  // Get the results
  _copyOutputBufferToMat(currentLayer->errors);
}

void VulkanController::_computeShader(const NeuronMat &neurons,
                                      VkCommandBuffer &commandBuffer,
                                      VkPipeline &pipeline) {
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkan_->pipelineLayout, 0, 1,
                          &vulkan_->descriptorSet, 0, nullptr);
  uint32_t rows = static_cast<uint32_t>(neurons.size());
  uint32_t cols = rows > 0 ? static_cast<uint32_t>(neurons[0].size()) : 0;
  vkCmdDispatch(commandBuffer, cols, rows, 1);
}

VkCommandBuffer VulkanController::_beginSingleTimeCommands() {
  // Take a command buffer from the pool
  VkCommandBuffer commandBuffer = vulkan_->commandBufferPool.back();
  vulkan_->commandBufferPool.pop_back();

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  // Starts recording the command
  auto result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan command buffer start error.");
  }
  return commandBuffer;
}

void VulkanController::_endSingleTimeCommands(VkCommandBuffer &commandBuffer) {
  // Ends recording the command
  auto result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan command buffer end error.");
  }

  // Submit the command to the queue
  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  result = vkQueueSubmit(vulkan_->queue, 1, &submitInfo, vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan queue submit error.");
  }
  // Wait for the fence to signal that the GPU has finished
  result = vkWaitForFences(vulkan_->logicalDevice, 1, &vulkan_->computeFence,
                           VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan wait for fence error.");
  }

  // Reset the fence
  result = vkResetFences(vulkan_->logicalDevice, 1, &vulkan_->computeFence);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan reset fence error.");
  }

  // Reset the command buffer
  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    throw VulkanControllerException("Vulkan reset command buffer error.");
  }

  vulkan_->commandBufferPool.push_back(commandBuffer);
}

void VulkanController::_copyNeuronsToBuffer(const NeuronMat &neurons,
                                            Buffer &buffer) {
  size_t totalNeuronsSize = 0;
  for (const auto &row : neurons) {
    totalNeuronsSize += row.size();
  }

  std::vector<GLSLNeuron> flatNeurons;
  flatNeurons.reserve(totalNeuronsSize);
  for (const auto &row : neurons) {
    for (const auto &neuron : row) {
      flatNeurons.push_back({.index_x = (uint)neuron.index_x,
                             .index_y = (uint)neuron.index_y,
                             .weightsIndex = (uint)neuron.weightsIndex,
                             .neighborsIndex = (uint)neuron.neighborsIndex,
                             .neighborsSize = (uint)neuron.neighborsSize});
    }
  }
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatNeurons.data(),
         (size_t)flatNeurons.size() * sizeof(GLSLNeuron));
}

void VulkanController::_copyMatToBuffer(const cv::Mat &mat, Buffer &buffer) {
  std::vector<cv::Vec4f> flatValues;
  flatValues.reserve(mat.total());
  for (int y = 0; y < mat.rows; y++) {
    for (int x = 0; x < mat.cols; x++) {
      flatValues.push_back(mat.at<cv::Vec4f>(x, y));
    }
  }
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatValues.data(),
         (size_t)flatValues.size() * sizeof(cv::Vec4f));
}

void VulkanController::_copyParametersToParametersBuffer(Layer *currentLayer) {
  const auto &network_params = Manager::getConstInstance().network_params;
  GLSLParameters parameters{
      .error_min = network_params.error_min,
      .error_max = network_params.error_max,
      .activationAlpha = currentLayer->activationFunctionAlpha,
      .currentLayerSizeX = (uint)currentLayer->size_x,
      .currentLayerSizeY = (uint)currentLayer->size_y,
      .previousLayerSizeX = currentLayer->previousLayer
                                ? (uint)currentLayer->previousLayer->size_x
                                : 0,
      .previousLayerSizeY = currentLayer->previousLayer
                                ? (uint)currentLayer->previousLayer->size_y
                                : 0,
      .nextLayerSizeX =
          currentLayer->nextLayer ? (uint)currentLayer->nextLayer->size_x : 0,
      .nextLayerSizeY =
          currentLayer->nextLayer ? (uint)currentLayer->nextLayer->size_y : 0,
      .activationFunction = (uint)currentLayer->eactivationFunction};
  auto &buffer = getBuffer(EBuffer::Parameters);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &parameters, sizeof(GLSLParameters));
}

// Flatten the all the weights vectors
void VulkanController::_copyNeuronsWeightsToWeightsBuffer(
    const NeuronMat &neurons) {
  // Get total sum of weights of every neurons
  size_t totalWeightsSize = 0;
  for (const auto &row : neurons) {
    totalWeightsSize += std::accumulate(row.begin(), row.end(), 0ull,
                                        [](size_t sum, const Neuron &neuron) {
                                          return sum + neuron.weights.total();
                                        });
  }
  // flatten every weights mat to a flatWeights vector
  std::vector<cv::Vec4f> flatWeights;
  flatWeights.reserve(totalWeightsSize);
  size_t weightsIndex = 0;
  for (const auto &row : neurons) {
    for (auto &neuron : row) {
      neuron.weightsIndex = weightsIndex;
      neuron.weights.forEach<cv::Vec4f>(
          [&flatWeights](const auto &value, const int *pos) {
            flatWeights.push_back(value);
          });
      weightsIndex += neuron.weights.total();
    }
  }
  // copy the flatWeights to the buffer
  auto &buffer = getBuffer(EBuffer::LayerWeights);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, flatWeights.data(),
         flatWeights.size() * sizeof(cv::Vec4f));
}

void VulkanController::_copyNeuronNeighboorsConnectionToBuffer(Layer *layer) {
  // get the neighboors connections weights and erros
  std::vector<cv::Vec4f> neighboorsConnectionWeights;
  std::vector<cv::Vec4f> neighboorsConnectionErrors;
  size_t connectionWeightsIndex = 0;
  for (const auto &row : layer->neurons) {
    for (auto &neuron : row) {
      neuron.neighborsSize = neuron.neighbors.size();
      neuron.neighborsIndex = connectionWeightsIndex;
      for (auto &connection : neuron.neighbors) {
        neighboorsConnectionWeights.push_back(connection.weight);
        neighboorsConnectionErrors.push_back(layer->errors.at<cv::Vec4f>(
            (int)connection.neuron->index_x, (int)connection.neuron->index_y));
      }
      connectionWeightsIndex += neuron.neighborsSize;
    }
  }
  // copy the weights to the buffer
  auto &bufferWeights = getBuffer(EBuffer::CurrentNeighborsWeights);
  memset(bufferWeights.data, 0, (size_t)bufferWeights.info.size);
  memcpy(bufferWeights.data, neighboorsConnectionWeights.data(),
         neighboorsConnectionWeights.size() * sizeof(cv::Vec4f));
  // copy the errors to the buffer
  auto &bufferErrors = getBuffer(EBuffer::CurrentNeighborsErrors);
  memset(bufferErrors.data, 0, (size_t)bufferErrors.info.size);
  memcpy(bufferErrors.data, neighboorsConnectionErrors.data(),
         neighboorsConnectionErrors.size() * sizeof(cv::Vec4f));
}

// Copy the OutputBuffer data directly into the value field of the neurons
void VulkanController::_copyOutputBufferToMat(cv::Mat &mat) {
  auto &bufferOutput = getBuffer(EBuffer::Output);
  // retrieve the data
  const auto bufferDataArray =
      static_cast<std::array<float, 4> *>(bufferOutput.data);

  // copy the data
  size_t index = 0;
  for (int y = 0; y < mat.rows; y++) {
    for (int x = 0; x < mat.cols; x++) {
      mat.at<cv::Vec4f>(x, y) =
          cv::Vec4f(bufferDataArray[index][0], bufferDataArray[index][1],
                    bufferDataArray[index][2], bufferDataArray[index][3]);
      index++;
    }
  }
  // some debug logs
  const auto &app_params = Manager::getConstInstance().app_params;
  if (app_params.verbose_debug) {
    std::atomic<bool> areAllZero = true;
    mat.forEach<cv::Vec4f>([&areAllZero](const auto &value, const int *pos) {
      if (value != cv::Vec4f::all(0.0)) {
        areAllZero = false;
      }
    });
    if (areAllZero) {
      SimpleLogger::LOG_DEBUG("Warning: all matrix values are zero.");
    }
  }
}
