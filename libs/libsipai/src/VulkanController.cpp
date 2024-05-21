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
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vulkan/vulkan.hpp>

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
      .withVulkan(vulkan_)
      .build();

  return vulkan_->isInitialized;
}

float VulkanController::trainingMonitored(
    const std::shared_ptr<sipai::Image> &inputValues,
    const std::shared_ptr<sipai::Image> &targetValues,
    const TrainingPhase &phase) {
  if (!IsInitialized()) {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }
  auto &trainingMonitoredShader = getShader(EShader::TrainingMonitored);

  _copyParameters();

  if (!trainingMonitoredShader.isReady) {
    _copyInputLayer();
    _copyOutputLayer();
    _copyHiddenLayer1();
    trainingMonitoredShader.isReady = true;
  }

  // Inject input data
  _copyInputData(inputValues->data, targetValues->data,
                 phase == TrainingPhase::Validation);

  // Run the shader
  _computeShader(trainingMonitoredShader.pipeline);

  // Get the results
  const auto result = _getOutputData();

  return result->loss;
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

void VulkanController::_copyParameters() {
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
  if (sizeof(GLSLInputLayer) > (size_t)buffer.info.size) {
    throw VulkanControllerException("copy buffer overflow");
  }
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslInputLayer, sizeof(GLSLInputLayer));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyOutputLayer() {

  struct _Attribs {
    float activation_alpha;
    uint activation_function;
    uint size_x;
    uint size_y;
  };

  const auto &layers = Manager::getConstInstance().network->layers;
  const auto &outputLayer = layers.back();
  if (outputLayer->layerType != LayerType::LayerOutput) {
    throw VulkanControllerException("Invalid Output layer type.");
  }

  // Copy the layer into the VRAM
  try {
    auto &buffer = getBuffer(EBuffer::OutputLayer);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    size_t totalSize = 0;
    char *bufferPtr = static_cast<char *>(buffer.data);

    // Copy the neurons
    for (const auto &row : outputLayer->neurons) {
      for (const auto &neuron : row) {
        size_t size = sizeof(uint);
        memcpy(bufferPtr, &neuron.index_x, size);
        bufferPtr += size;
        totalSize += size;

        size = sizeof(uint);
        memcpy(bufferPtr, &neuron.index_y, size);
        bufferPtr += size;
        totalSize += size;

        size = neuron.weights.total() * sizeof(cv::Vec4f);
        memcpy(bufferPtr, neuron.weights.data, size);
        bufferPtr += size;
        totalSize += size;

        GLSLNeighbor neighbors[MAX_NEIGHBORS];
        for (int i = 0; i < neuron.neighbors.size(); i++) {
          neighbors[i].index_x = (uint)neuron.neighbors[i].neuron->index_x;
          neighbors[i].index_y = (uint)neuron.neighbors[i].neuron->index_y;
          neighbors[i].weight = neuron.neighbors[i].weight;
          neighbors[i].is_used = true;
        }
        size = MAX_NEIGHBORS * sizeof(GLSLNeighbor);
        memcpy(bufferPtr, &neighbors, size);
        bufferPtr += size;
        totalSize += size;
      }
    }

    // Copy the errors
    size_t size = outputLayer->errors.total() * sizeof(cv::Vec4f);
    memcpy(bufferPtr, outputLayer->errors.data, size);
    bufferPtr += size;
    totalSize += size;

    // Copy the attributes
    _Attribs attribs{
        .activation_alpha = outputLayer->activationFunctionAlpha,
        .activation_function = (uint)outputLayer->eactivationFunction,
        .size_x = (uint)outputLayer->size_x,
        .size_y = (uint)outputLayer->size_y,
    };
    size = sizeof(_Attribs);
    memcpy(bufferPtr, &attribs, size);
    bufferPtr += size;
    totalSize += size;

    builder_.unmapBufferMemory(buffer);

    if (totalSize > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Hidden layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_copyHiddenLayer1() {

  struct _Attribs {
    float activation_alpha;
    uint activation_function;
    uint size_x;
    uint size_y;
  };

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
    size_t totalSize = 0;
    char *bufferPtr = static_cast<char *>(buffer.data);

    // Copy the neurons
    for (const auto &row : hiddenLayer1->neurons) {
      for (const auto &neuron : row) {
        // index_x
        size_t size = sizeof(uint);
        memcpy(bufferPtr, &neuron.index_x, size);
        bufferPtr += size;
        totalSize += size;

        // index_y
        size = sizeof(uint);
        memcpy(bufferPtr, &neuron.index_y, size);
        bufferPtr += size;
        totalSize += size;

        // weights
        size = neuron.weights.total() * sizeof(cv::Vec4f);
        memcpy(bufferPtr, neuron.weights.data, size);
        bufferPtr += size;
        totalSize += size;

        // neighbors
        GLSLNeighbor neighbors[MAX_NEIGHBORS];
        for (int i = 0; i < neuron.neighbors.size(); i++) {
          neighbors[i].index_x = (uint)neuron.neighbors[i].neuron->index_x;
          neighbors[i].index_y = (uint)neuron.neighbors[i].neuron->index_y;
          neighbors[i].weight = neuron.neighbors[i].weight;
          neighbors[i].is_used = true;
        }
        size = MAX_NEIGHBORS * sizeof(GLSLNeighbor);
        memcpy(bufferPtr, &neighbors, size);
        bufferPtr += size;
        totalSize += size;
      }
    }

    // Copy the values
    size_t size = hiddenLayer1->values.total() * sizeof(cv::Vec4f);
    memcpy(bufferPtr, hiddenLayer1->values.data, size);
    bufferPtr += size;
    totalSize += size;

    // Copy the errors
    size = hiddenLayer1->errors.total() * sizeof(cv::Vec4f);
    memcpy(bufferPtr, hiddenLayer1->errors.data, size);
    bufferPtr += size;
    totalSize += size;

    // Copy the attributes
    _Attribs attribs{
        .activation_alpha = hiddenLayer1->activationFunctionAlpha,
        .activation_function = (uint)hiddenLayer1->eactivationFunction,
        .size_x = (uint)hiddenLayer1->size_x,
        .size_y = (uint)hiddenLayer1->size_y,
    };
    size = sizeof(_Attribs);
    memcpy(bufferPtr, &attribs, size);
    bufferPtr += size;
    totalSize += size;

    builder_.unmapBufferMemory(buffer);

    if (totalSize > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Hidden layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_copyInputData(const cv::Mat &inputValues,
                                      const cv::Mat &targetValues,
                                      bool is_validation) {
  // Copy the data into the VRAM
  try {
    auto &buffer = getBuffer(EBuffer::InputData);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    char *bufferPtr = static_cast<char *>(buffer.data);
    size_t totalSize = 0;

    size_t size = inputValues.total() * sizeof(cv::Vec4f);
    memcpy(bufferPtr, inputValues.data, size);
    bufferPtr += size;
    totalSize += size;

    size = targetValues.total() * sizeof(cv::Vec4f);
    memcpy(bufferPtr, targetValues.data, size);
    bufferPtr += size;
    totalSize += size;

    size = sizeof(bool);
    memcpy(bufferPtr, &is_validation, size);
    bufferPtr += size;
    totalSize += size;

    builder_.unmapBufferMemory(buffer);

    if (totalSize > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Input data copy error: " +
                                    std::string(ex.what()));
  }
}

std::unique_ptr<GLSLOutputData> VulkanController::_getOutputData() {
  const auto &params = Manager::getConstInstance().network_params;

  // Get loss
  auto &bufferLoss = getBuffer(EBuffer::OutputLoss);
  builder_.mapBufferMemory(bufferLoss);
  float loss = *reinterpret_cast<float *>(bufferLoss.data);
  builder_.unmapBufferMemory(bufferLoss);

  // Get outputValues
  // Commented: not required here
  // cv::Mat outputValues((int)params.output_size_y,
  // (int)params.output_size_x,
  //                      CV_32FC4, cv::Vec4f::all(0.0));
  // auto &buffer = getBuffer(EBuffer::OutputValues);
  // builder_.mapBufferMemory(buffer);
  // const auto mappedData = static_cast<std::array<float, 4> *>(buffer.data);
  // // copy the data
  // for (int y = 0; y < outputValues.rows; y++) {
  //   for (int x = 0; x < outputValues.cols; x++) {
  //     size_t index = y * outputValues.cols + x;
  //     outputValues.at<cv::Vec4f>(y, x) =
  //         cv::Vec4f(mappedData[index][0], mappedData[index][1],
  //                   mappedData[index][2], mappedData[index][3]);
  //   }
  // }
  // builder_.unmapBufferMemory(buffer);
  return std::make_unique<GLSLOutputData>(GLSLOutputData{.loss = loss});
}
