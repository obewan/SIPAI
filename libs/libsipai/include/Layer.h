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
#include "Neuron.h"
#include <map>

namespace sipai {
enum class LayerType { InputLayer, HiddenLayer, OutputLayer };

const std::map<std::string, LayerType, std::less<>> layer_map{
    {"InputLayer", LayerType::InputLayer},
    {"HiddenLayer", LayerType::HiddenLayer},
    {"OutputLayer", LayerType::OutputLayer}};

/**
 * @brief The Layer class represents a layer in a neural network. It contains a
 * vector of Neurons and has methods for forward propagation, backward
 * propagation, and updating weights.
 */
class Layer {
public:
  explicit Layer(LayerType layerType) : layerType(layerType) {}

  const LayerType layerType;
  std::vector<Neuron> neurons;
  Layer *previousLayer = nullptr;
  Layer *nextLayer = nullptr;

  const std::string UndefinedLayer = "UndefinedLayer";

  // Virtual destructor
  virtual ~Layer() = default;

  /**
   * @brief Performs forward propagation using the previous layer.
   *
   */
  virtual void forwardPropagation() = 0;

  /**
   * @brief Performs backward propagation using the next layer.
   *
   */
  virtual void backwardPropagation() = 0;

  /**
   * @brief Updates the weights of the neurons in this layer using the previous
   * layer and a learning rate.
   *
   * @param learningRate The learning rate to use when updating weights.
   */
  virtual void updateWeights(float learningRate) = 0;

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