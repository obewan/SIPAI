/**
 * @file VulkanController.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Vulkan Controller
 * @date 2024-04-19
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "Layer.h"
#include "VulkanBuilder.h"
#include "VulkanCommon.h"
#include "VulkanHelper.h"
#include "exception/VulkanControllerException.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace sipai
{
  class VulkanController
  {
  public:
    static VulkanController &getInstance()
    {
      static std::once_flag initInstanceFlag;
      std::call_once(initInstanceFlag,
                     []
                     { controllerInstance_.reset(new VulkanController); });
      return *controllerInstance_;
    }
    static const VulkanController &getConstInstance()
    {
      return const_cast<const VulkanController &>(getInstance());
    }
    VulkanController(VulkanController const &) = delete;
    void operator=(VulkanController const &) = delete;
    ~VulkanController() { destroy(); }

    bool initialize();

    const bool IsInitialized() { return vulkan_->isInitialized; }

    /**
     * @brief Vulkan training or validation on an input image
     *
     *  @return float computed loss between the generated output and the expected
     * images after the training or the validation
     */
    float training(const std::shared_ptr<sipai::Image> &inputValues,
                   const std::shared_ptr<sipai::Image> &targetValues,
                   const TrainingPhase &phase);

    /**
     * @brief Get image output values after a forward propagation, using an
     * existing neural network.
     *
     * @param inputValues the input values
     */
    void forwardEnhancer(const cv::Mat &inputValues);

    /**
     * @brief Update the Neural Network with values from Vulkan.
     *
     */
    void updateNeuralNetwork();

    /**
     * @brief Destroy the device instance, cleaning ressources
     *
     */
    void destroy() { builder_.clear(); };

    /**
     * @brief Get the Logical Device
     *
     * @return VkDevice&
     */
    VkDevice &getDevice() { return vulkan_->logicalDevice; }

    /**
     * @brief Get a Buffer
     *
     * @param bufferName
     * @return Buffer&
     */
    Buffer &getBuffer(const EBuffer &bufferName)
    {
      if (!vulkan_)
      {
        throw VulkanControllerException("vulkan is null pointer.");
      }
      auto it = std::find_if(
          vulkan_->buffers.begin(), vulkan_->buffers.end(),
          [&bufferName](auto &buffer)
          { return buffer.name == bufferName; });
      if (it != vulkan_->buffers.end())
      {
        return *it;
      }
      else
      {
        throw VulkanControllerException("buffer not found.");
      }
    }

    Shader &getShader(const EShader &shaderName)
    {
      if (!vulkan_)
      {
        throw VulkanControllerException("vulkan is null pointer.");
      }
      auto it = std::find_if(vulkan_->shaders.begin(), vulkan_->shaders.end(),
                             [&shaderName](auto &shader)
                             {
                               return shader.shadername == shaderName;
                             });
      if (it != vulkan_->shaders.end())
      {
        return *it;
      }
      else
      {
        throw VulkanControllerException("shader not found.");
      }
    }

    VkPipeline getPipelineForShader(const EShader &shaderName)
    {
      auto it = vulkan_->computePipelines.find(shaderName);
      if (it != vulkan_->computePipelines.end())
      {
        return it->second;
      }
      else
      {
        throw std::runtime_error("Pipeline for the specified shader not found");
      }
    }

    std::shared_ptr<Vulkan> getVulkan() { return vulkan_; }

    /**
     * @brief Helper function to copy data from the buffer
     *
     * @tparam T
     * @param bufferData
     * @param offset
     * @return T
     */
    template <typename T>
    T getDataFromBuffer(const void *bufferData, uint32_t &offset)
    {
      const T *typedBufferData = reinterpret_cast<const T *>(bufferData);
      T value = *(typedBufferData + offset / sizeof(T));
      offset += sizeof(T);
      return value;
    }

    /**
     * @brief Set the Data To Buffer
     *
     * @tparam T
     * @param buffer
     * @param data
     * @return size_t buffer new position
     */
    template <typename T>
    uint8_t *copyToBuffer(uint8_t *buffer, const T &data)
    {
      memcpy(buffer, &data, sizeof(T));
      return buffer + sizeof(T);
    }

    /**
     * @brief Swap Endianess
     *
     * @param value
     * @return uint32_t
     */
    uint32_t swapEndian(uint32_t value)
    {
      return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) |
             ((value & 0x00FF0000) >> 8) | ((value & 0xFF000000) >> 24);
    }

  private:
    VulkanController()
    {
      vulkan_ = std::make_shared<Vulkan>();
      helper_.setVulkan(vulkan_);
    };
    static std::unique_ptr<VulkanController> controllerInstance_;

    void _processShaders(const EShader &shader);
    void _processRenderPass(VkCommandBuffer &commandBuffer);
    void _readHiddenLayer1();
    void _readOutputLayer();
    void _readOutputData();
    float _readOutputLoss();

    void _writeParameters();
    void _writeInputLayer();
    void _writeOutputLayer();
    void _writeHiddenLayer1();
    void _writeInputData(const cv::Mat &inputValues);
    void _writeInputData(const cv::Mat &inputValues, const cv::Mat &targetValues,
                         const TrainingPhase &phase);

    std::shared_ptr<Vulkan> vulkan_;
    VulkanBuilder builder_;
    VulkanHelper helper_;
  };
} // namespace sipai
