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
#include <cstddef>
#include <execution>
#include <functional>
#include <map>

namespace sipai {
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

  const std::string UndefinedLayer = "UndefinedLayer";

  /**
   * @brief Performs forward propagation using the previous layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void forwardPropagation(bool enable_parallel = false) {
    if (previousLayer == nullptr) {
      return;
    }
    if (enable_parallel) {
      _forward(std::execution::par_unseq);
    } else {
      _forward(std::execution::seq);
    }
  };

  /**
   * @brief Performs backward propagation using the next layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void backwardPropagation(bool enable_parallel = false) {
    if (nextLayer == nullptr) {
      return;
    }
    if (enable_parallel) {
      _backward(std::execution::par_unseq);
    } else {
      _backward(std::execution::seq);
    }
  }

  /**
   * @brief Updates the weights of the neurons in this layer using the previous
   * layer and a learning rate.
   *
   * @param learningRate The learning rate to use when updating weights.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void updateWeights(float learningRate, bool enable_parallel = false) {
    if (previousLayer == nullptr) {
      return;
    }
    if (enable_parallel) {
      _update(std::execution::par_unseq, learningRate);
    } else {
      _update(std::execution::seq, learningRate);
    }
  }

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

protected:
  /**
   * @brief Forward propagation core methode
   *
   * @tparam ExecutionPolicy the sequential or parallel execution policy type
   * @param executionPolicy
   */
  template <typename ExecutionPolicy>
  void _forward(ExecutionPolicy executionPolicy) {
    if (previousLayer == nullptr) {
      return;
    }
    std::for_each(executionPolicy, neurons.begin(), neurons.end(),
                  [this](Neuron &n) {
                    n.value = {0.0, 0.0, 0.0, 0.0};
                    for (size_t i = 0; i < previousLayer->neurons.size(); i++) {
                      n.value += previousLayer->neurons[i].value * n.weights[i];
                    }

                    // Use activation function
                    n.value = n.activationFunction(n.value);
                  });
  };

  /**
   * @brief Backward propagation core methode
   *
   * @tparam ExecutionPolicy
   * @param executionPolicy
   */
  template <typename ExecutionPolicy>
  void _backward(ExecutionPolicy executionPolicy) {
    if (nextLayer == nullptr) {
      return;
    }
    float error_min = -1.0f;
    float error_max = 1.0f;
    std::for_each(executionPolicy, neurons.begin(), neurons.end(),
                  [this, &error_min, &error_max](Neuron &n) {
                    size_t pos = &n - &neurons[0];
                    RGBA error = {0.0, 0.0, 0.0, 0.0};
                    for (Neuron &nn : nextLayer->neurons) {
                      error += nn.weights[pos] * nn.error;
                    }

                    // Consider errors of adjacent neurons
                    for (NeuronConnection &conn : n.neighbors) {
                      error += conn.neuron->error * conn.weight;
                    }
                    // Use the derivative of the activation function
                    n.error = (error * n.activationFunctionDerivative(n.value))
                                  .clamp(error_min, error_max);
                  });
  }

  /**
   * @brief Update weights core methode
   *
   * @tparam ExecutionPolicy the sequential or parallel execution policy type
   * @param executionPolicy
   * @param learningRate
   */
  template <typename ExecutionPolicy>
  void _update(ExecutionPolicy executionPolicy, float learningRate) {
    if (previousLayer == nullptr) {
      return;
    }
    float error_min = -1.0f;
    float error_max = 1.0f;
    std::for_each(executionPolicy, neurons.begin(), neurons.end(),
                  [this, &learningRate, &error_min, &error_max](Neuron &n) {
                    // Update weights based on neurons in the previous layer
                    for (size_t j = 0; j < n.weights.size(); ++j) {
                      auto dE_dw = previousLayer->neurons[j].value * n.error;
                      n.weights[j] -=
                          learningRate * dE_dw.clamp(error_min, error_max);
                      n.weights[j] = n.weights[j].clamp();
                    }

                    // Update weights based on neighboring neurons
                    for (NeuronConnection &conn : n.neighbors) {
                      auto dE_dw = conn.neuron->value * n.error;
                      conn.weight -=
                          learningRate * dE_dw.clamp(error_min, error_max);
                    }
                  });
  };
};
} // namespace sipai