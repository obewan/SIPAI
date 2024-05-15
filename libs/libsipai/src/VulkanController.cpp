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
      .build(vulkan_);

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
  if (!trainingMonitoredShader.isReady) {
    _copyParameters();
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
  memcpy(buffer.data, &glslInputLayer, sizeof(glslInputLayer));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyOutputLayer() {

  const auto &layers = Manager::getConstInstance().network->layers;
  const auto &outputLayer = layers.back();
  if (outputLayer->layerType != LayerType::LayerOutput) {
    throw VulkanControllerException("Invalid Output layer type.");
  }

  // Create the layer neurons, including their weights
  std::vector<std::vector<GLSLNeuron>> glslNeurons(
      outputLayer->size_y, std::vector<GLSLNeuron>(outputLayer->size_x));
  for (int y = 0; y < outputLayer->neurons.size(); y++) {
    for (int x = 0; x < outputLayer->neurons[0].size(); x++) {
      const auto &neuron = outputLayer->neurons[y][x];
      GLSLNeuron glslNeuron = {.index_x = (uint)neuron.index_x,
                               .index_y = (uint)neuron.index_y};
      glslNeuron.weights = std::vector<std::vector<cv::Vec4f>>(
          neuron.weights.rows, std::vector<cv::Vec4f>(neuron.weights.cols));
      Common::copyMatToVector(neuron.weights, glslNeuron.weights);

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

  // Create the GLSL formatted layer
  GLSLOutputLayer glslOutputLayer{
      .neurons = glslNeurons,
      .errors = std::vector<std::vector<cv::Vec4f>>(
          outputLayer->errors.rows,
          std::vector<cv::Vec4f>(outputLayer->errors.cols)),
      .activation_alpha = outputLayer->activationFunctionAlpha,
      .activation_function = (uint)outputLayer->eactivationFunction,
      .size_x = (uint)outputLayer->size_x,
      .size_y = (uint)outputLayer->size_y,
  };
  Common::copyMatToVector(outputLayer->errors, glslOutputLayer.errors);

  // Copy the layer into the VRAM
  auto &buffer = getBuffer(EBuffer::OutputLayer);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslOutputLayer, sizeof(glslOutputLayer));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyHiddenLayer1() {
  const auto &layers = Manager::getConstInstance().network->layers;
  if (layers.size() < 2) {
    throw VulkanControllerException("Invalid layers size.");
  }
  const auto &hiddenLayer1 = layers.at(1);
  if (hiddenLayer1->layerType != LayerType::LayerHidden) {
    throw VulkanControllerException("Invalid Hidden layer type.");
  }

  // Create the layer neurons, including their weights
  std::vector<std::vector<GLSLNeuron>> glslNeurons(
      hiddenLayer1->size_y, std::vector<GLSLNeuron>(hiddenLayer1->size_x));
  for (int y = 0; y < hiddenLayer1->neurons.size(); y++) {
    for (int x = 0; x < hiddenLayer1->neurons[0].size(); x++) {
      const auto &neuron = hiddenLayer1->neurons[y][x];
      GLSLNeuron glslNeuron = {.index_x = (uint)neuron.index_x,
                               .index_y = (uint)neuron.index_y};
      glslNeuron.weights = std::vector<std::vector<cv::Vec4f>>(
          neuron.weights.rows, std::vector<cv::Vec4f>(neuron.weights.cols));
      Common::copyMatToVector(neuron.weights, glslNeuron.weights);

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

  // Create the GLSL formatted layer
  GLSLHiddenLayer glslHiddenLayer{
      .neurons = glslNeurons,
      .values = std::vector<std::vector<cv::Vec4f>>(
          hiddenLayer1->values.rows,
          std::vector<cv::Vec4f>(hiddenLayer1->values.cols)),
      .errors = std::vector<std::vector<cv::Vec4f>>(
          hiddenLayer1->errors.rows,
          std::vector<cv::Vec4f>(hiddenLayer1->errors.cols)),
      .activation_alpha = hiddenLayer1->activationFunctionAlpha,
      .activation_function = (uint)hiddenLayer1->eactivationFunction,
      .size_x = (uint)hiddenLayer1->size_x,
      .size_y = (uint)hiddenLayer1->size_y,
  };
  Common::copyMatToVector(hiddenLayer1->values, glslHiddenLayer.values);
  Common::copyMatToVector(hiddenLayer1->errors, glslHiddenLayer.errors);

  // Copy the layer into the VRAM
  auto &buffer = getBuffer(EBuffer::HiddenLayer1);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslHiddenLayer, sizeof(glslHiddenLayer));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_copyInputData(const cv::Mat &inputValues,
                                      const cv::Mat &targetValues,
                                      bool is_validation) {
  // Create the GLSL formatted data
  GLSLInputData glslInputData{
      .inputValues = std::vector<std::vector<cv::Vec4f>>(
          inputValues.rows, std::vector<cv::Vec4f>(inputValues.cols)),
      .targetValues = std::vector<std::vector<cv::Vec4f>>(
          targetValues.rows, std::vector<cv::Vec4f>(targetValues.cols)),
      .is_validation = is_validation};
  Common::copyMatToVector(inputValues, glslInputData.inputValues);
  Common::copyMatToVector(targetValues, glslInputData.targetValues);

  // Copy the data into the VRAM
  auto &buffer = getBuffer(EBuffer::InputData);
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslInputData, sizeof(glslInputData));
  builder_.unmapBufferMemory(buffer);
}

std::unique_ptr<GLSLOutputData> VulkanController::_getOutputData() {
  auto &buffer = getBuffer(EBuffer::OutputData);
  builder_.mapBufferMemory(buffer);
  std::unique_ptr<GLSLOutputData> data = std::make_unique<GLSLOutputData>(
      *static_cast<GLSLOutputData *>(buffer.data));
  builder_.unmapBufferMemory(buffer);
  return data;
}
