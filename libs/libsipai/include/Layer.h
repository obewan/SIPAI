/**
 * @file Layer.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Abstract layer class
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "ActivationFunctions.h"
#include "Neuron.h"
#include "exception/NeuralNetworkException.h"
#include <atomic>
#include <cstddef>
#include <execution>
#include <functional>
#include <map>
#include <thread>

namespace sipai {
class VulkanController;

enum class LayerType { LayerInput, LayerHidden, LayerOutput };

const std::map<std::string, LayerType, std::less<>> layer_map{
    {"LayerInput", LayerType::LayerInput},
    {"LayerHidden", LayerType::LayerHidden},
    {"LayerOutput", LayerType::LayerOutput}};

/**
 * @brief The Layer class represents a layer in a neural network. It contains a
 * vector of Neurons and has methods for forward propagation, backward
 * propagation, and updating weights.
 */
class Layer {
public:
  explicit Layer(LayerType layerType, size_t size_x = 0, size_t size_y = 0)
      : layerType(layerType), size_x(size_x), size_y(size_y) {}
  virtual ~Layer() = default;

  const LayerType layerType;
  std::vector<Neuron> neurons;
  Layer *previousLayer = nullptr;
  Layer *nextLayer = nullptr;
  size_t size_x = 0;
  size_t size_y = 0;
  EActivationFunction activationFunction;
  float activationFunctionAlpha;

  const std::string UndefinedLayer = "UndefinedLayer";

  /**
   * @brief Performs forward propagation using the previous layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void forwardPropagation(bool enable_vulkan = false,
                                  bool enable_parallel = false);

  /**
   * @brief Performs backward propagation using the next layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void backwardPropagation(const float &error_min,
                                   const float &error_max,
                                   bool enable_parallel = false);
  /**
   * @brief Updates the weights of the neurons in this layer using the
   * previous layer and a learning rate.
   *
   * @param learningRate The learning rate to use when updating weights.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void updateWeights(float learningRate, bool enable_parallel = false);

  const std::string getLayerTypeStr() const {
    for (const auto &[key, mLayerType] : layer_map) {
      if (mLayerType == layerType) {
        return key;
      }
    }
    return UndefinedLayer;
  }

  void setActivationFunction(const std::function<RGBA(RGBA)> &function,
                             const std::function<RGBA(RGBA)> &derivative) {
    for (auto &n : neurons) {
      n.activationFunction = function;
      n.activationFunctionDerivative = derivative;
    }
  }
};
} // namespace sipai