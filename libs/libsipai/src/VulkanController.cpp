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
  if (!helper_.replaceTemplateParameters(
          manager.app_params.trainingMonitoredShaderTemplate,
          manager.app_params.trainingMonitoredShader)) {
    SimpleLogger::LOG_ERROR("Templated shader build error.");
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

float VulkanController::trainingMonitored(const TrainingPhase &phase) {
  if (!IsInitialized()) {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }
  auto &trainingMonitoredShader = getShader(EShader::TrainingMonitored);
  if (!trainingMonitoredShader.isReady) {
    _copyParameters();
    _copyInputLayer();

    trainingMonitoredShader.isReady = true;
  }
  // TODO continue...
  return 0.0;
}

void VulkanController::_computeShader(VkPipeline &pipeline) {
  auto commandBuffer = helper_.beginSingleTimeCommands();
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          vulkan_->pipelineLayout, 0, 1,
                          &vulkan_->descriptorSet, 0, nullptr);
  vkCmdDispatch(commandBuffer, 1, 1, 1);
  helper_.endSingleTimeCommands(commandBuffer);
}

void VulkanController::_copyNeuronsToBuffer(const NeuronMat &neurons,
                                            Buffer &buffer) {
  // size_t totalNeuronsSize = 0;
  // for (const auto &row : neurons) {
  //   totalNeuronsSize += row.size();
  // }

  // std::vector<GLSLNeuron> flatNeurons;
  // flatNeurons.reserve(totalNeuronsSize);
  // for (const auto &row : neurons) {
  //   for (const auto &neuron : row) {
  //     flatNeurons.push_back({.index_x = (uint)neuron.index_x,
  //                            .index_y = (uint)neuron.index_y,
  //                            .weightsIndex = (uint)neuron.weightsIndex,
  //                            .neighborsIndex = (uint)neuron.neighborsIndex,
  //                            .neighborsSize = (uint)neuron.neighborsSize});
  //   }
  // }
  // memset(buffer.data, 0, (size_t)buffer.info.size);
  // memcpy(buffer.data, flatNeurons.data(),
  //        (size_t)flatNeurons.size() * sizeof(GLSLNeuron));
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

void VulkanController::_copyParameters() {
  const auto &network_params = Manager::getConstInstance().network_params;
  GLSLParameters glslParams{
      .learning_rate = network_params.learning_rate,
      .error_min = network_params.error_min,
      .error_max = network_params.error_max,
  };
  auto &buffer = getBuffer(EBuffer::Parameters);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslParams, sizeof(GLSLParameters));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyInputLayer() {
  const auto &inputLayer = Manager::getConstInstance().network->layers.front();
  if (inputLayer->layerType != LayerType::LayerInput) {
    throw VulkanControllerException("Invalid Input layer type.");
  }
  GLSLInputLayer glslInputLayer{
      .activation_alpha = inputLayer->activationFunctionAlpha,
      .activation_function = (uint)inputLayer->eactivationFunction,
      .size_x = (uint)inputLayer->size_x,
      .size_y = (uint)inputLayer->size_y,
  };
  auto &buffer = getBuffer(EBuffer::InputLayer);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslInputLayer, sizeof(GLSLInputLayer));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyOutputLayer() {
  auto copyMat = [](const cv::Mat &mat,
                    std::vector<std::vector<cv::Vec4f>> &array) {
    for (int i = 0; i < mat.rows; ++i) {
      for (int j = 0; j < mat.cols; ++j) {
        array[i][j] = mat.at<cv::Vec4f>(i, j);
      }
    }
  };

  const auto &layers = Manager::getConstInstance().network->layers;
  const auto &outputLayer = layers.back();
  if (outputLayer->layerType != LayerType::LayerOutput) {
    throw VulkanControllerException("Invalid Output layer type.");
  }

  struct GLSLOutputNeuron {
    uint index_x;
    uint index_y;
    std::vector<std::vector<cv::Vec4f>> weights;
    GLSLNeighbor neighbors[4];
  };
  struct GLSLOutputLayer {
    std::vector<std::vector<GLSLOutputNeuron>> neurons;
    std::vector<std::vector<cv::Vec4f>> errors;
    float activation_alpha;
    uint activation_function;
    uint size_x;
    uint size_y;
  };

  std::vector<std::vector<GLSLOutputNeuron>> glslNeurons(
      outputLayer->size_y, std::vector<GLSLOutputNeuron>(outputLayer->size_x));
  for (int y = 0; y < outputLayer->neurons.size(); y++) {
    for (int x = 0; x < outputLayer->neurons[0].size(); x++) {
      const auto &neuron = outputLayer->neurons[y][x];
      GLSLOutputNeuron glslNeuron = {.index_x = (uint)neuron.index_x,
                                     .index_y = (uint)neuron.index_y};
      glslNeuron.weights = std::vector<std::vector<cv::Vec4f>>(
          neuron.weights.rows, std::vector<cv::Vec4f>(neuron.weights.cols));
      copyMat(neuron.weights, glslNeuron.weights);

      for (int i = 0; i < neuron.neighbors.size() && i < 4; i++) {
        glslNeuron.neighbors[i].index_x =
            (uint)neuron.neighbors[i].neuron->index_x;
        glslNeuron.neighbors[i].index_y =
            (uint)neuron.neighbors[i].neuron->index_y;
        glslNeuron.neighbors[i].weight = neuron.neighbors[i].weight;
        glslNeuron.neighbors[i].is_used = true;
      }
      glslNeurons[y][x] = glslNeuron;
    }
  }
  GLSLOutputLayer glslOutputLayer{.neurons = glslNeurons};
  glslOutputLayer.errors = std::vector<std::vector<cv::Vec4f>>(
      outputLayer->errors.rows,
      std::vector<cv::Vec4f>(outputLayer->errors.cols));
  copyMat(outputLayer->errors, glslOutputLayer.errors);
  auto &buffer = getBuffer(EBuffer::OutputLayer);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslOutputLayer, sizeof(GLSLOutputLayer));
  builder_.unmapBufferMemory(buffer);
}

// Flatten the all the weights vectors
void VulkanController::_copyNeuronsWeightsToWeightsBuffer(
    const NeuronMat &neurons) {
  // // Get total sum of weights of every neurons
  // size_t totalWeightsSize = 0;
  // for (const auto &row : neurons) {
  //   totalWeightsSize += std::accumulate(row.begin(), row.end(), 0ull,
  //                                       [](size_t sum, const Neuron &neuron)
  //                                       {
  //                                         return sum +
  //                                         neuron.weights.total();
  //                                       });
  // }
  // // flatten every weights mat to a flatWeights vector
  // std::vector<cv::Vec4f> flatWeights;
  // flatWeights.reserve(totalWeightsSize);
  // size_t weightsIndex = 0;
  // for (const auto &row : neurons) {
  //   for (auto &neuron : row) {
  //     neuron.weightsIndex = weightsIndex;
  //     neuron.weights.forEach<cv::Vec4f>(
  //         [&flatWeights](const auto &value, const int *pos) {
  //           flatWeights.push_back(value);
  //         });
  //     weightsIndex += neuron.weights.total();
  //   }
  // }
  // // copy the flatWeights to the buffer
  // auto &buffer = getBuffer(EBuffer::LayerWeights);
  // memset(buffer.data, 0, (size_t)buffer.info.size);
  // memcpy(buffer.data, flatWeights.data(),
  //        flatWeights.size() * sizeof(cv::Vec4f));
}

void VulkanController::_copyNeuronNeighboorsConnectionToBuffer(Layer *layer) {
  // // get the neighboors connections weights and erros
  // std::vector<cv::Vec4f> neighboorsConnectionWeights;
  // std::vector<cv::Vec4f> neighboorsConnectionErrors;
  // size_t connectionWeightsIndex = 0;
  // for (const auto &row : layer->neurons) {
  //   for (auto &neuron : row) {
  //     neuron.neighborsSize = neuron.neighbors.size();
  //     neuron.neighborsIndex = connectionWeightsIndex;
  //     for (auto &connection : neuron.neighbors) {
  //       neighboorsConnectionWeights.push_back(connection.weight);
  //       neighboorsConnectionErrors.push_back(layer->errors.at<cv::Vec4f>(
  //           (int)connection.neuron->index_x,
  //           (int)connection.neuron->index_y));
  //     }
  //     connectionWeightsIndex += neuron.neighborsSize;
  //   }
  // }
  // // copy the weights to the buffer
  // auto &bufferWeights = getBuffer(EBuffer::CurrentNeighborsWeights);
  // memset(bufferWeights.data, 0, (size_t)bufferWeights.info.size);
  // memcpy(bufferWeights.data, neighboorsConnectionWeights.data(),
  //        neighboorsConnectionWeights.size() * sizeof(cv::Vec4f));
  // // copy the errors to the buffer
  // auto &bufferErrors = getBuffer(EBuffer::CurrentNeighborsErrors);
  // memset(bufferErrors.data, 0, (size_t)bufferErrors.info.size);
  // memcpy(bufferErrors.data, neighboorsConnectionErrors.data(),
  //        neighboorsConnectionErrors.size() * sizeof(cv::Vec4f));
}

// Copy the OutputBuffer data directly into the value field of the neurons
void VulkanController::_copyOutputBufferToMat(cv::Mat &mat) {
  // auto &bufferOutput = getBuffer(EBuffer::Output);
  // // retrieve the data
  // const auto bufferDataArray =
  //     static_cast<std::array<float, 4> *>(bufferOutput.data);

  // // copy the data
  // size_t index = 0;
  // for (int x = 0; x < mat.rows; x++) {
  //   for (int y = 0; y < mat.cols; y++) {
  //     mat.at<cv::Vec4f>(x, y) =
  //         cv::Vec4f(bufferDataArray[index][0], bufferDataArray[index][1],
  //                   bufferDataArray[index][2], bufferDataArray[index][3]);
  //     index++;
  //   }
  // }
  // // some debug logs
  // const auto &app_params = Manager::getConstInstance().app_params;
  // if (app_params.verbose_debug) {
  //   std::atomic<bool> areAllZero = true;
  //   mat.forEach<cv::Vec4f>([&areAllZero](const auto &value, const int *pos) {
  //     if (value != cv::Vec4f::all(0.0)) {
  //       areAllZero = false;
  //     }
  //   });
  //   if (areAllZero) {
  //     SimpleLogger::LOG_DEBUG("Warning: all matrix values are zero.");
  //   }
  // }
}
