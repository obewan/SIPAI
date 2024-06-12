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

// Specialization for uint32_t
template <>
uint32_t VulkanController::getDataFromBuffer<uint32_t>(const void *bufferData,
                                                       uint32_t &offset) {
  const uint32_t *typedBufferData =
      reinterpret_cast<const uint32_t *>(bufferData);
  uint32_t value = *(typedBufferData + offset / sizeof(uint32_t));
  // value = swapEndian(value);
  offset += sizeof(uint32_t);
  return value;
}

// Specialization for uint32_t
template <>
uint8_t *VulkanController::copyToBuffer<uint32_t>(uint8_t *buffer,
                                                  const uint32_t &data) {
  // uint32_t swap = swapEndian(data);
  // memcpy(buffer, &swap, sizeof(uint32_t));
  memcpy(buffer, &data, sizeof(uint32_t));
  return buffer + sizeof(uint32_t);
}

bool VulkanController::initialize() {
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
    // templated shaders
    if (!helper_.replaceTemplateParameters(
            manager.app_params.shaderTrainingMonitoredTemplate,
            manager.app_params.shaderTrainingMonitored)) {
      SimpleLogger::LOG_ERROR("Templated shader build error.");
      return false;
    }

    // shaders list
    vulkan_->shaders.push_back(
        {.shadername = EShader::TrainingMonitoredShader,
         .filename = manager.app_params.shaderTrainingMonitored});
    vulkan_->shaders.push_back({.shadername = EShader::VertexShader,
                                .filename = manager.app_params.shaderVertex});
    vulkan_->shaders.push_back({.shadername = EShader::FragmentShader,
                                .filename = manager.app_params.shaderFragment});
  }

  // initialize opencv window (before builder)
  // !!! OpenCV must be built with OpenGL support
  // https://answers.opencv.org/question/10592/opencv-error-no-opengl-support/
  cv::namedWindow(cvWindowTitle, (int)cv::WINDOW_OPENGL);
  cv::resizeWindow(cvWindowTitle, 800, 600);
  cv::imshow(cvWindowTitle, cv::Mat::zeros(600, 800, CV_8UC3));

  // Vulkan builder
  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(manager.app_params.verbose_debug)
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
  auto &trainingMonitoredShader = getShader(EShader::TrainingMonitoredShader);

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
  _computeShader(vulkan_->pipelineCompute);

  // Get the results
  const auto result = _getOutputData();

  return result->loss;
}

void VulkanController::updateNeuralNetwork() {
  _readBackHiddenLayer1();
  _readBackOutputLayer();
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

void VulkanController::_readBackHiddenLayer1() {
  auto &network = Manager::getInstance().network;
  if (!network || network->layers.size() < 2 ||
      network->layers.at(1)->layerType != LayerType::LayerHidden) {
    throw VulkanControllerException("invalid neural network");
  }
  auto &hiddenLayer = network->layers.at(1);
  auto &bufferHiddenLayer = getBuffer(EBuffer::HiddenLayer1);

  builder_.mapBufferMemory(bufferHiddenLayer);
  if (!bufferHiddenLayer.data) {
    builder_.unmapBufferMemory(bufferHiddenLayer);
    throw VulkanControllerException(
        "Invalid data pointer after mapping buffer memory");
  }

  uint offset = 0;

  // Read neurons
  for (size_t y = 0; y < hiddenLayer->size_y; ++y) {
    for (size_t x = 0; x < hiddenLayer->size_x; ++x) {
      auto &dstNeuron = hiddenLayer->neurons[y][x];

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
    for (int y = 0; y < hiddenLayer->values.rows; ++y) {
      for (int x = 0; x < hiddenLayer->values.cols; ++x) {
        auto value =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        hiddenLayer->values.at<cv::Vec4f>(y, x) = value;
      }
    }

    // Get errors
    for (int y = 0; y < hiddenLayer->errors.rows; ++y) {
      for (int x = 0; x < hiddenLayer->errors.cols; ++x) {
        auto error =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        hiddenLayer->errors.at<cv::Vec4f>(y, x) = error;
      }
    }

    // Get others attributes (offset update)
    getDataFromBuffer<float>(bufferHiddenLayer.data,
                             offset); // activation_alpha
    getDataFromBuffer<uint>(bufferHiddenLayer.data,
                            offset); // activation_function
    getDataFromBuffer<uint>(bufferHiddenLayer.data, offset); // size_x
    getDataFromBuffer<uint>(bufferHiddenLayer.data, offset); // size_y
  } // End for Read neurons

  builder_.unmapBufferMemory(bufferHiddenLayer);
}

void VulkanController::_readBackOutputLayer() {
  auto &network = Manager::getInstance().network;
  if (!network || network->layers.size() < 2 ||
      network->layers.back()->layerType != LayerType::LayerOutput) {
    throw VulkanControllerException("invalid neural network");
  }
  // auto &outputLayer = network->layers.back();

  // auto &bufferOutputLayer = getBuffer(EBuffer::OutputLayer);

  // TODO readBackOutputLayer()
  // builder_.mapBufferMemory(bufferOutputLayer);
  // builder_.unmapBufferMemory(bufferOutputLayer);
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

  try {
    auto &buffer = getBuffer(EBuffer::InputLayer);

    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;
    bufferPtr =
        copyToBuffer<float>(bufferPtr, inputLayer->activationFunctionAlpha);
    bufferPtr = copyToBuffer<uint32_t>(
        bufferPtr, (uint32_t)inputLayer->eactivationFunction);
    bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)inputLayer->size_x);
    bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)inputLayer->size_y);
    builder_.unmapBufferMemory(buffer);

    size_t totalBytesCopied = bufferPtr - bufferStart;
    if (totalBytesCopied > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Input layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_copyOutputLayer() {
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
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the neurons
    for (const auto &row : outputLayer->neurons) {
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
            bufferPtr =
                copyToBuffer<bool>(bufferPtr, static_cast<uint32_t>(isUsed));
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            for (int k = 0; k < 4; k++) {
              bufferPtr = copyToBuffer<float>(bufferPtr, 0.0f);
            }
          }
        }
      }
    }

    // Copy the errors
    for (int y = 0; y < outputLayer->errors.rows; y++) {
      for (int x = 0; x < outputLayer->errors.cols; x++) {
        for (int k = 0; k < 4; k++) {
          bufferPtr = copyToBuffer<float>(
              bufferPtr, outputLayer->errors.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the attributes
    bufferPtr =
        copyToBuffer<float>(bufferPtr, outputLayer->activationFunctionAlpha);
    bufferPtr = copyToBuffer<uint32_t>(
        bufferPtr, (uint32_t)outputLayer->eactivationFunction);
    bufferPtr =
        copyToBuffer<uint32_t>(bufferPtr, (uint32_t)outputLayer->size_x);
    bufferPtr =
        copyToBuffer<uint32_t>(bufferPtr, (uint32_t)outputLayer->size_y);

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

void VulkanController::_copyHiddenLayer1() {
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
            bufferPtr =
                copyToBuffer<bool>(bufferPtr, static_cast<uint32_t>(isUsed));
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

void VulkanController::_copyInputData(const cv::Mat &inputValues,
                                      const cv::Mat &targetValues,
                                      bool is_validation) {
  // Copy the data into the VRAM
  try {
    auto &buffer = getBuffer(EBuffer::InputData);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the inputValues
    for (int y = 0; y < inputValues.rows; y++) {
      for (int x = 0; x < inputValues.cols; x++) {
        for (int k = 0; k < 4; k++) {
          bufferPtr = copyToBuffer<float>(bufferPtr,
                                          inputValues.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the targetValues
    for (int y = 0; y < targetValues.rows; y++) {
      for (int x = 0; x < targetValues.cols; x++) {
        for (int k = 0; k < 4; k++) {
          bufferPtr = copyToBuffer<float>(bufferPtr,
                                          targetValues.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy is_validation
    bufferPtr = copyToBuffer<bool>(bufferPtr, is_validation);

    builder_.unmapBufferMemory(buffer);

    size_t totalBytesCopied = bufferPtr - bufferStart;
    if (totalBytesCopied > (size_t)buffer.info.size) {
      throw VulkanControllerException("copy buffer overflow");
    }
  } catch (std::exception &ex) {
    throw VulkanControllerException("Input data copy error: " +
                                    std::string(ex.what()));
  }
}

std::unique_ptr<GLSLOutputData> VulkanController::_getOutputData() {
  GLSLOutputData outputData = {};

  // Get loss
  auto &bufferLoss = getBuffer(EBuffer::OutputLoss);
  builder_.mapBufferMemory(bufferLoss);
  outputData.loss = *reinterpret_cast<float *>(bufferLoss.data);
  builder_.unmapBufferMemory(bufferLoss);

  // Get outputValues
  // Commented: not required here
  // const auto &params = Manager::getConstInstance().network_params;
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

  return std::make_unique<GLSLOutputData>(outputData);
}
