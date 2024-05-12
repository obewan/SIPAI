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

bool VulkanController::initialize(bool enableDebug) {
  if (vulkan_->isInitialized) {
    return true;
  }

  const auto &manager = Manager::getConstInstance();
  if (!manager.network) {
    return false;
  }

  vulkan_->shaders.clear();
  if (!helper_.replaceTemplateParameters(
          manager.app_params.trainingMonitoredShaderTemplate,
          manager.app_params.trainingMonitoredShader)) {
    return false;
  }
  vulkan_->shaders.push_back(
      {.shadername = EShader::TrainingMonitored,
       .filename = manager.app_params.trainingMonitoredShader});

  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(enableDebug)
      .build(vulkan_);

  return vulkan_->isInitialized;
}

void VulkanController::trainingMonitored() {
  // TODO
}

void VulkanController::_computeShader(const NeuronMat &neurons,
                                      VkPipeline &pipeline) {
  auto commandBuffer = helper_.beginSingleTimeCommands();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkan_->pipelineLayout, 0, 1,
                          &vulkan_->descriptorSet, 0, nullptr);
  uint32_t rows = static_cast<uint32_t>(neurons.size());
  uint32_t cols = rows > 0 ? static_cast<uint32_t>(neurons[0].size()) : 0;
  vkCmdDispatch(commandBuffer, cols, rows, 1);
  helper_.endSingleTimeCommands(commandBuffer);
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
  for (int x = 0; x < mat.rows; x++) {
    for (int y = 0; y < mat.cols; y++) {
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
  for (int x = 0; x < mat.rows; x++) {
    for (int y = 0; y < mat.cols; y++) {
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
