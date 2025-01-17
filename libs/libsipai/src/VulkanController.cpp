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

bool VulkanController::initialize()
{
  if (vulkan_->isInitialized)
  {
    return true;
  }

  const auto &manager = Manager::getConstInstance();
  if (!manager.network)
  {
    SimpleLogger::LOG_ERROR("No neural network found.");
    return false;
  }

  if (manager.network->layers.size() != 3)
  {
    SimpleLogger::LOG_ERROR(
        "The current Vulkan shader is limited to exactly 3 layers : an input "
        "layer, a hidden layer and an output layer.");
    return false;
  }

  vulkan_->maxSizeX =
      std::max({manager.network_params.input_size_x, manager.network_params.hidden_size_x,
                manager.network_params.output_size_x});
  vulkan_->maxSizeY =
      std::max({manager.network_params.input_size_y, manager.network_params.hidden_size_y,
                manager.network_params.output_size_y});

  if (vulkan_->shaders.empty())
  {
    // templated shaders
    for (auto &shader : manager.app_params.shaders)
    {
      if (!shader.templateFilename.empty() && !helper_.replaceTemplateParameters(
                                                  shader.templateFilename, shader.filename))
      {
        SimpleLogger::LOG_ERROR("Templated shader build error.");
        return false;
      }
      vulkan_->shaders.push_back({.shadername = shader.name,
                                  .filename = shader.filename});
    }
  }

  // initialize opencv window (before builder)
  // !!! OpenCV must be built with OpenGL support
  // https://answers.opencv.org/question/10592/opencv-error-no-opengl-support/
  if (manager.app_params.vulkan_debug)
  {
    cv::namedWindow(cvWindowTitle, (int)cv::WINDOW_OPENGL);
    cv::resizeWindow(cvWindowTitle, (int)vulkan_->window_width,
                     (int)vulkan_->window_height);
    cv::imshow(cvWindowTitle, cv::Mat::zeros((int)vulkan_->window_height,
                                             (int)vulkan_->window_width,
                                             CV_8UC3)); // rows, cols, type
  }

  // Vulkan builder
  builder_.withCommandPoolSize(1)
      .withMaxNeighboorsPerNeuron(4)
      .withDebugInfo(manager.app_params.verbose_debug)
      .withVulkan(vulkan_)
      .build();

  return vulkan_->isInitialized;
}

float VulkanController::training(
    const std::shared_ptr<sipai::Image> &inputValues,
    const std::shared_ptr<sipai::Image> &targetValues,
    const TrainingPhase &phase)
{
  if (!IsInitialized())
  {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }
  auto &trainingShader = getShader(EShader::TrainingInit);

  _writeParameters();

  if (!trainingShader.isReady)
  {
    _writeInputLayer();
    _writeOutputLayer();
    _writeHiddenLayer1();
    trainingShader.isReady = true;
  }

  // Inject input data
  _writeInputData(inputValues->data, targetValues->data, phase);

  // Compute (draw 3D frame if vulkan debug, can be debug in RenderDoc then)
  _processShaders(EShader::TrainingInit);

  // Get the results
  const float loss = _readOutputLoss();

  return loss;
}

void VulkanController::forwardEnhancer(const cv::Mat &inputValues)
{
  if (!IsInitialized())
  {
    throw VulkanControllerException("Vulkan controller is not initialized.");
  }

  auto &enhancerShader = getShader(EShader::EnhancerForward1);

  _writeParameters();

  if (!enhancerShader.isReady)
  {
    _writeInputLayer();
    _writeOutputLayer();
    _writeHiddenLayer1();
    enhancerShader.isReady = true;
  }

  // Inject input data
  _writeInputData(inputValues);

  // Compute (draw 3D frame if vulkan debug, can be debug in RenderDoc then)
  _processShaders(EShader::EnhancerForward1);

  // Get the results into the output layer values
  _readOutputData();
}

void VulkanController::updateNeuralNetwork()
{
  _readHiddenLayer1();
  _readOutputLayer();
}

/**
 * @brief if vulkan debug mode, draw a window to be used with RenderDoc with
 * pauses, else do just a compute shader pass.
 *
 */
void VulkanController::_processShaders(const EShader &shader)
{
  const auto &app_params = Manager::getConstInstance().app_params;

  // Begin recording commands in a single-time command buffer
  auto commandBuffer = helper_.commandsBegin();

  // Compute pass begin
  switch (shader)
  {
  case EShader::TrainingInit:
  {
    // compute each training steps with a barrier between each
    std::list<EShader> shaderStages = {
        EShader::TrainingInit, EShader::TrainingForward1, EShader::TrainingForward2,
        EShader::TrainingForward3, EShader::TrainingForward4,
        EShader::TrainingBackward1, EShader::TrainingBackward2,
        EShader::TrainingBackward3, EShader::TrainingBackward4};

    for (const auto &shaderName : shaderStages)
    {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        getPipelineForShader(shaderName));
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              vulkan_->pipelineLayout, 0, 1,
                              &vulkan_->descriptorSet, 0, nullptr);
      vkCmdDispatch(commandBuffer, (int)vulkan_->maxSizeX, (int)vulkan_->maxSizeY, 1);

      // Insert a pipeline barrier to ensure proper synchronization
      VkMemoryBarrier memoryBarrier = {};
      memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Source stage
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Destination stage
                           0,                                    // Dependency flags
                           1,                                    // Memory barrier count
                           &memoryBarrier,                       // Memory barriers
                           0,                                    // Buffer memory barrier count
                           nullptr,                              // Buffer memory barriers
                           0,                                    // Image memory barrier count
                           nullptr);                             // Image memory barriers
    }
    break;
  }
  case EShader::EnhancerForward1:
  {
    // compute each training steps with a barrier between each
    std::list<EShader> shaderStages = {EShader::EnhancerForward1, EShader::EnhancerForward2};

    for (const auto &shaderName : shaderStages)
    {
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                        getPipelineForShader(shaderName));
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                              vulkan_->pipelineLayout, 0, 1,
                              &vulkan_->descriptorSet, 0, nullptr);
      vkCmdDispatch(commandBuffer, (int)vulkan_->maxSizeX, (int)vulkan_->maxSizeY, 1);

      // Insert a pipeline barrier to ensure proper synchronization
      VkMemoryBarrier memoryBarrier = {};
      memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

      vkCmdPipelineBarrier(commandBuffer,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Source stage
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, // Destination stage
                           0,                                    // Dependency flags
                           1,                                    // Memory barrier count
                           &memoryBarrier,                       // Memory barriers
                           0,                                    // Buffer memory barrier count
                           nullptr,                              // Buffer memory barriers
                           0,                                    // Image memory barrier count
                           nullptr);                             // Image memory barriers
    }
    break;
  }
  default:
    throw VulkanControllerException("Non implemented compute shader");
  }
  // Compute pass end

  if (app_params.vulkan_debug)
  {
    // If vulkan debug, using graphic pipeline and render pass in window
    _processRenderPass(commandBuffer);
  }
  else
  {
    // else just submit and return
    helper_.commandsEnd_SubmitQueueCompute(commandBuffer);
  }
}

void VulkanController::_processRenderPass(VkCommandBuffer &commandBuffer)
{

  uint32_t imageIndex;

  // Wait for the fence to ensure the previous frame is finished
  auto result = vkWaitForFences(vulkan_->logicalDevice, 1,
                                &vulkan_->inFlightFence, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS)
  {
    throw VulkanControllerException("Failed to wait for fence");
  }

  // Acquire an image from the swap chain
  result = vkAcquireNextImageKHR(vulkan_->logicalDevice, vulkan_->swapChain,
                                 UINT64_MAX, vulkan_->imageAvailableSemaphore,
                                 VK_NULL_HANDLE, &imageIndex);
  if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
  {
    throw VulkanControllerException("Failed to acquire swap chain image");
  }

  // Memory barrier to ensure the compute pass is finished before starting the
  // render pass
  VkMemoryBarrier memoryBarrier = {};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

  // Set up the render pass begin info
  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = vulkan_->renderPass;
  renderPassInfo.framebuffer = vulkan_->swapChainFramebuffers[imageIndex];
  renderPassInfo.renderArea.offset = {0, 0};
  renderPassInfo.renderArea.extent = vulkan_->swapChainExtent;

  VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearColor;

  // Begin the render pass and bind the pipeline
  vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                       VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    vulkan_->graphicPipeline);

  // Bind the vertex buffer
  auto &vertexBuffer = getBuffer(EBuffer::Vertex);
  VkBuffer vertexBuffers[] = {vertexBuffer.buffer};
  VkDeviceSize offsets[] = {0};
  vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

  // Issue the draw command
  vkCmdDraw(commandBuffer, static_cast<uint32_t>(vulkan_->vertices.size()), 1,
            0, 0);

  // End the render pass
  vkCmdEndRenderPass(commandBuffer);

  // End recording commands and queue submit
  helper_.commandsEnd_SubmitQueueGraphics(commandBuffer, imageIndex);
}

void VulkanController::_readHiddenLayer1()
{
  auto &network = Manager::getInstance().network;
  if (!network || network->layers.size() < 2 ||
      network->layers.at(1)->layerType != LayerType::LayerHidden)
  {
    throw VulkanControllerException("invalid neural network");
  }
  auto &hiddenLayer = network->layers.at(1);
  auto &bufferHiddenLayer = getBuffer(EBuffer::HiddenLayer1);

  builder_.mapBufferMemory(bufferHiddenLayer);
  if (!bufferHiddenLayer.data)
  {
    builder_.unmapBufferMemory(bufferHiddenLayer);
    throw VulkanControllerException(
        "Invalid data pointer after mapping buffer memory");
  }

  uint offset = 0;

  // Read neurons
  for (size_t y = 0; y < hiddenLayer->size_y; ++y)
  {
    for (size_t x = 0; x < hiddenLayer->size_x; ++x)
    {
      auto &dstNeuron = hiddenLayer->neurons[y][x];

      // Check index_x and index_y
      uint32_t index_x =
          getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
      uint32_t index_y =
          getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
      if (dstNeuron.index_x != index_x || dstNeuron.index_y != index_y)
      {
        builder_.unmapBufferMemory(bufferHiddenLayer);
        throw VulkanControllerException("Invalid data buffer memory");
      }

      // Get weights
      for (int i = 0; i < dstNeuron.weights.rows; ++i)
      {
        for (int j = 0; j < dstNeuron.weights.cols; ++j)
        {
          auto value = cv::Vec4f(
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
              getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
          dstNeuron.weights.at<cv::Vec4f>(i, j) = value;
        }
      }

      // Get neighbors
      int neighbors_padding = 8; // check with RenderDoc
      offset += neighbors_padding;
      for (int i = 0; i < MAX_NEIGHBORS; i++)
      {
        uint32_t isUsed =
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
        uint32_t neigh_index_x =
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);
        uint32_t neigh_index_y =
            getDataFromBuffer<uint32_t>(bufferHiddenLayer.data, offset);

        // Some checks
        if (isUsed &&
            (dstNeuron.neighbors[i].neuron->index_x != neigh_index_x ||
             dstNeuron.neighbors[i].neuron->index_y != neigh_index_y))
        {
          builder_.unmapBufferMemory(bufferHiddenLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }
        if (((isUsed > 0) && (i + 1 > (int)dstNeuron.neighbors.size())) ||
            ((isUsed <= 0) && (i + 1 < (int)dstNeuron.neighbors.size())))
        {
          builder_.unmapBufferMemory(bufferHiddenLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }

        // Get connection weight
        offset += 4; // padding, check with RenderDoc
        auto weight =
            cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                      getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
        if (isUsed > 0)
        {
          dstNeuron.neighbors[i].weight = weight;
        }
      }
    } // end for (size_t x ...
  } // end for (size_t y ...

  // Get values
  for (int y = 0; y < hiddenLayer->values.rows; ++y)
  {
    for (int x = 0; x < hiddenLayer->values.cols; ++x)
    {
      auto value =
          cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
      hiddenLayer->values.at<cv::Vec4f>(y, x) = value;
    }
  }

  // Get errors
  for (int y = 0; y < hiddenLayer->errors.rows; ++y)
  {
    for (int x = 0; x < hiddenLayer->errors.cols; ++x)
    {
      auto error =
          cv::Vec4f(getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset),
                    getDataFromBuffer<float>(bufferHiddenLayer.data, offset));
      hiddenLayer->errors.at<cv::Vec4f>(y, x) = error;
    }
  }

  // Get others attributes
  // Check activation_alpha and activation_function
  auto activation_alpha =
      getDataFromBuffer<float>(bufferHiddenLayer.data, offset);
  auto activation_function =
      getDataFromBuffer<uint>(bufferHiddenLayer.data, offset);
  float epsilon = 0.0001f;
  if (abs(activation_alpha - hiddenLayer->activationFunctionAlpha) > epsilon ||
      activation_function != (uint32_t)hiddenLayer->eactivationFunction)
  {
    builder_.unmapBufferMemory(bufferHiddenLayer);
    throw VulkanControllerException("Invalid data buffer memory");
  }
  // Check size_x and size_y
  auto size_x = getDataFromBuffer<uint>(bufferHiddenLayer.data, offset);
  auto size_y = getDataFromBuffer<uint>(bufferHiddenLayer.data, offset);
  if (size_x != hiddenLayer->size_x || size_y != hiddenLayer->size_y)
  {
    builder_.unmapBufferMemory(bufferHiddenLayer);
    throw VulkanControllerException("Invalid data buffer memory");
  }
  builder_.unmapBufferMemory(bufferHiddenLayer);
}

void VulkanController::_readOutputLayer()
{
  auto &network = Manager::getInstance().network;
  if (!network || network->layers.size() < 2 ||
      network->layers.back()->layerType != LayerType::LayerOutput)
  {
    throw VulkanControllerException("invalid neural network");
  }
  auto &outputLayer = network->layers.back();
  auto &bufferOutputLayer = getBuffer(EBuffer::OutputLayer);

  builder_.mapBufferMemory(bufferOutputLayer);
  if (!bufferOutputLayer.data)
  {
    builder_.unmapBufferMemory(bufferOutputLayer);
    throw VulkanControllerException(
        "Invalid data pointer after mapping buffer memory");
  }

  uint offset = 0;

  // Read neurons
  for (size_t y = 0; y < outputLayer->size_y; ++y)
  {
    for (size_t x = 0; x < outputLayer->size_x; ++x)
    {
      auto &dstNeuron = outputLayer->neurons[y][x];

      // Check index_x and index_y
      uint32_t index_x =
          getDataFromBuffer<uint32_t>(bufferOutputLayer.data, offset);
      uint32_t index_y =
          getDataFromBuffer<uint32_t>(bufferOutputLayer.data, offset);
      if (dstNeuron.index_x != index_x || dstNeuron.index_y != index_y)
      {
        builder_.unmapBufferMemory(bufferOutputLayer);
        throw VulkanControllerException("Invalid data buffer memory");
      }

      // Get weights
      for (int i = 0; i < dstNeuron.weights.rows; ++i)
      {
        for (int j = 0; j < dstNeuron.weights.cols; ++j)
        {
          auto value = cv::Vec4f(
              getDataFromBuffer<float>(bufferOutputLayer.data, offset),
              getDataFromBuffer<float>(bufferOutputLayer.data, offset),
              getDataFromBuffer<float>(bufferOutputLayer.data, offset),
              getDataFromBuffer<float>(bufferOutputLayer.data, offset));
          dstNeuron.weights.at<cv::Vec4f>(i, j) = value;
        }
      }

      // Get neighbors
      int neighbors_padding = 8; // check with RenderDoc
      offset += neighbors_padding;
      for (int i = 0; i < MAX_NEIGHBORS; i++)
      {
        uint32_t isUsed =
            getDataFromBuffer<uint32_t>(bufferOutputLayer.data, offset);
        uint32_t neigh_index_x =
            getDataFromBuffer<uint32_t>(bufferOutputLayer.data, offset);
        uint32_t neigh_index_y =
            getDataFromBuffer<uint32_t>(bufferOutputLayer.data, offset);

        // Some checks
        if (isUsed &&
            (dstNeuron.neighbors[i].neuron->index_x != neigh_index_x ||
             dstNeuron.neighbors[i].neuron->index_y != neigh_index_y))
        {
          builder_.unmapBufferMemory(bufferOutputLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }
        if (((isUsed > 0) && (i + 1 > (int)dstNeuron.neighbors.size())) ||
            ((isUsed <= 0) && (i + 1 < (int)dstNeuron.neighbors.size())))
        {
          builder_.unmapBufferMemory(bufferOutputLayer);
          throw VulkanControllerException("Invalid data buffer memory");
        }

        // Get connection weight
        offset += 4; // padding, check with RenderDoc
        auto weight =
            cv::Vec4f(getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                      getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                      getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                      getDataFromBuffer<float>(bufferOutputLayer.data, offset));
        if (isUsed > 0)
        {
          dstNeuron.neighbors[i].weight = weight;
        }
      }
    } // end for (size_t x ...
  } // end for (size_t y ...

  // Get errors
  for (int y = 0; y < outputLayer->errors.rows; ++y)
  {
    for (int x = 0; x < outputLayer->errors.cols; ++x)
    {
      auto error =
          cv::Vec4f(getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                    getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                    getDataFromBuffer<float>(bufferOutputLayer.data, offset),
                    getDataFromBuffer<float>(bufferOutputLayer.data, offset));
      outputLayer->errors.at<cv::Vec4f>(y, x) = error;
    }
  }

  // Get others attributes
  // Check activation_alpha and activation_function
  auto activation_alpha =
      getDataFromBuffer<float>(bufferOutputLayer.data, offset);
  auto activation_function =
      getDataFromBuffer<uint>(bufferOutputLayer.data, offset);
  float epsilon = 0.0001f;
  if (abs(activation_alpha - outputLayer->activationFunctionAlpha) > epsilon ||
      activation_function != (uint32_t)outputLayer->eactivationFunction)
  {
    builder_.unmapBufferMemory(bufferOutputLayer);
    throw VulkanControllerException("Invalid data buffer memory");
  }
  // Check size_x and size_y
  auto size_x = getDataFromBuffer<uint>(bufferOutputLayer.data, offset);
  auto size_y = getDataFromBuffer<uint>(bufferOutputLayer.data, offset);
  if (size_x != outputLayer->size_x || size_y != outputLayer->size_y)
  {
    builder_.unmapBufferMemory(bufferOutputLayer);
    throw VulkanControllerException("Invalid data buffer memory");
  }
  builder_.unmapBufferMemory(bufferOutputLayer);
}

float VulkanController::_readOutputLoss()
{
  auto &buffer = getBuffer(EBuffer::OutputLoss);

  builder_.mapBufferMemory(buffer);
  float loss = *reinterpret_cast<float *>(buffer.data);
  builder_.unmapBufferMemory(buffer);

  return loss;
}

void VulkanController::_readOutputData()
{

  auto &network = Manager::getInstance().network;
  auto &outputLayer = network->layers.back();

  try
  {
    uint offset = 0;
    auto &buffer = getBuffer(EBuffer::OutputData);
    builder_.mapBufferMemory(buffer);

    // Get output values
    for (int y = 0; y < outputLayer->values.rows; ++y)
    {
      for (int x = 0; x < outputLayer->values.cols; ++x)
      {
        auto value = cv::Vec4f(getDataFromBuffer<float>(buffer.data, offset),
                               getDataFromBuffer<float>(buffer.data, offset),
                               getDataFromBuffer<float>(buffer.data, offset),
                               getDataFromBuffer<float>(buffer.data, offset));
        outputLayer->values.at<cv::Vec4f>(y, x) = value;
      }
    }

    builder_.unmapBufferMemory(buffer);
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Reading Output Data error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_writeParameters()
{
  const auto &network_params = Manager::getConstInstance().network_params;
  GLSLParameters glslParams{
      .learning_rate = network_params.learning_rate,
      .error_min = network_params.error_min,
      .error_max = network_params.error_max,
  };
  auto &buffer = getBuffer(EBuffer::Parameters);
  if (sizeof(GLSLParameters) > (size_t)buffer.info.size)
  {
    throw VulkanControllerException("copy buffer overflow");
  }
  builder_.mapBufferMemory(buffer);
  memset(buffer.data, 0, (size_t)buffer.info.size);
  memcpy(buffer.data, &glslParams, sizeof(GLSLParameters));
  builder_.unmapBufferMemory(buffer);
}

void VulkanController::_writeInputLayer()
{
  const auto &inputLayer = Manager::getConstInstance().network->layers.front();
  if (inputLayer->layerType != LayerType::LayerInput)
  {
    throw VulkanControllerException("Invalid Input layer type.");
  }

  try
  {
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
    if (totalBytesCopied > (size_t)buffer.info.size)
    {
      throw VulkanControllerException("copy buffer overflow");
    }
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Input layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_writeOutputLayer()
{
  const auto &layers = Manager::getConstInstance().network->layers;
  const auto &outputLayer = layers.back();
  if (outputLayer->layerType != LayerType::LayerOutput)
  {
    throw VulkanControllerException("Invalid Output layer type.");
  }

  // Copy the layer into the VRAM
  try
  {
    auto &buffer = getBuffer(EBuffer::OutputLayer);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the neurons
    for (const auto &row : outputLayer->neurons)
    {
      for (const auto &neuron : row)
      {
        // index_xy
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_x);
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_y);

        // weights
        for (int y = 0; y < neuron.weights.rows; y++)
        {
          for (int x = 0; x < neuron.weights.cols; x++)
          {
            for (int k = 0; k < 4; k++)
            {
              float value = neuron.weights.at<cv::Vec4f>(y, x)[k];
              bufferPtr = copyToBuffer<float>(bufferPtr, value);
            }
          }
        }

        // neighbors
        int neighbors_padding = 8; // check with RenderDoc
        bufferPtr += neighbors_padding;
        uint32_t isUsed = 0;
        for (int i = 0; i < MAX_NEIGHBORS; i++)
        {
          if (i < (int)neuron.neighbors.size())
          {
            isUsed = 1;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, isUsed);
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_x);
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_y);
            bufferPtr += 4; // padding, check with RenderDoc
            for (int k = 0; k < 4; k++)
            {
              bufferPtr =
                  copyToBuffer<float>(bufferPtr, neuron.neighbors[i].weight[k]);
            }
          }
          else
          {
            isUsed = 0;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, isUsed);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr += 4; // padding, check with RenderDoc
            for (int k = 0; k < 4; k++)
            {
              bufferPtr = copyToBuffer<float>(bufferPtr, 0.0f);
            }
          }
        }
      }
    }

    // Copy the errors
    for (int y = 0; y < outputLayer->errors.rows; y++)
    {
      for (int x = 0; x < outputLayer->errors.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
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
    if (totalBytesCopied > (size_t)buffer.info.size)
    {
      throw VulkanControllerException("copy buffer overflow");
    }
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Hidden layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_writeHiddenLayer1()
{
  const auto &layers = Manager::getConstInstance().network->layers;
  if (layers.size() < 2)
  {
    throw VulkanControllerException("Invalid layers size.");
  }
  const auto &hiddenLayer1 = layers.at(1);
  if (hiddenLayer1->layerType != LayerType::LayerHidden)
  {
    throw VulkanControllerException("Invalid Hidden layer type.");
  }
  // Copy the layer into the VRAM
  try
  {
    auto &buffer = getBuffer(EBuffer::HiddenLayer1);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the neurons
    for (const auto &row : hiddenLayer1->neurons)
    {
      for (const auto &neuron : row)
      {
        // index_xy
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_x);
        bufferPtr = copyToBuffer<uint32_t>(bufferPtr, (uint32_t)neuron.index_y);

        // weights
        for (int y = 0; y < neuron.weights.rows; y++)
        {
          for (int x = 0; x < neuron.weights.cols; x++)
          {
            for (int k = 0; k < 4; k++)
            {
              float value = neuron.weights.at<cv::Vec4f>(y, x)[k];
              bufferPtr = copyToBuffer<float>(bufferPtr, value);
            }
          }
        }

        // neighbors
        int neighbors_padding = 8; // check with RenderDoc
        bufferPtr += neighbors_padding;
        uint32_t isUsed = 0;
        for (int i = 0; i < MAX_NEIGHBORS; i++)
        {
          if (i < (int)neuron.neighbors.size())
          {
            isUsed = 1;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, isUsed);
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_x);
            bufferPtr = copyToBuffer<uint32_t>(
                bufferPtr, (uint32_t)neuron.neighbors[i].neuron->index_y);
            bufferPtr += 4; // padding, check with RenderDoc
            for (int k = 0; k < 4; k++)
            {
              bufferPtr =
                  copyToBuffer<float>(bufferPtr, neuron.neighbors[i].weight[k]);
            }
          }
          else
          {
            isUsed = 0;
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, isUsed);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr = copyToBuffer<uint32_t>(bufferPtr, 0);
            bufferPtr += 4; // padding, check with RenderDoc
            for (int k = 0; k < 4; k++)
            {
              bufferPtr = copyToBuffer<float>(bufferPtr, 0.0f);
            }
          }
        }
      }
    }

    // Copy the values
    for (int y = 0; y < hiddenLayer1->values.rows; y++)
    {
      for (int x = 0; x < hiddenLayer1->values.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
          bufferPtr = copyToBuffer<float>(
              bufferPtr, hiddenLayer1->values.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the errors
    for (int y = 0; y < hiddenLayer1->errors.rows; y++)
    {
      for (int x = 0; x < hiddenLayer1->errors.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
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
    if (totalBytesCopied > (size_t)buffer.info.size)
    {
      throw VulkanControllerException("copy buffer overflow");
    }
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Hidden layer copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_writeInputData(const cv::Mat &inputValues)
{
  // Copy the data into the VRAM
  try
  {
    auto &buffer = getBuffer(EBuffer::InputData);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the inputValues
    for (int y = 0; y < inputValues.rows; y++)
    {
      for (int x = 0; x < inputValues.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
          bufferPtr = copyToBuffer<float>(bufferPtr,
                                          inputValues.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    size_t totalBytesCopied = bufferPtr - bufferStart;
    if (totalBytesCopied > (size_t)buffer.info.size)
    {
      throw VulkanControllerException("copy buffer overflow");
    }
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Input data copy error: " +
                                    std::string(ex.what()));
  }
}

void VulkanController::_writeInputData(const cv::Mat &inputValues,
                                       const cv::Mat &targetValues,
                                       const TrainingPhase &phase)
{
  // Copy the data into the VRAM
  try
  {
    auto &buffer = getBuffer(EBuffer::InputData);
    builder_.mapBufferMemory(buffer);
    memset(buffer.data, 0, (size_t)buffer.info.size);
    uint8_t *bufferPtr = static_cast<uint8_t *>(buffer.data);
    uint8_t *bufferStart = bufferPtr;

    // Copy the inputValues
    for (int y = 0; y < inputValues.rows; y++)
    {
      for (int x = 0; x < inputValues.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
          bufferPtr = copyToBuffer<float>(bufferPtr,
                                          inputValues.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy the targetValues
    for (int y = 0; y < targetValues.rows; y++)
    {
      for (int x = 0; x < targetValues.cols; x++)
      {
        for (int k = 0; k < 4; k++)
        {
          bufferPtr = copyToBuffer<float>(bufferPtr,
                                          targetValues.at<cv::Vec4f>(y, x)[k]);
        }
      }
    }

    // Copy is_validation
    bool is_validation = (phase == TrainingPhase::Validation);
    bufferPtr = copyToBuffer<uint>(bufferPtr, is_validation);

    builder_.unmapBufferMemory(buffer);

    size_t totalBytesCopied = bufferPtr - bufferStart;
    if (totalBytesCopied > (size_t)buffer.info.size)
    {
      throw VulkanControllerException("copy buffer overflow");
    }
  }
  catch (std::exception &ex)
  {
    throw VulkanControllerException("Input data copy error: " +
                                    std::string(ex.what()));
  }
}
