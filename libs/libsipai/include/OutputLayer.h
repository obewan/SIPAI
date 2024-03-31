/**
 * @file OutputLayer.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Output layer
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "Layer.h"
#include "Neuron.h"
#include <algorithm>
#include <cstddef>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace sipai {
/**
 * @brief The OutputLayer class represents the output layer of a neural network.
 * It inherits from the Layer class and overrides its methods as necessary. This
 * layer is responsible for producing the final output of the network.
 */
class OutputLayer : public Layer {
public:
  OutputLayer() : Layer(LayerType::OutputLayer) {}

  void forwardPropagation() override {
    if (previousLayer == nullptr) {
      return;
    }
    // Implement forward propagation for output layer
    for (auto &n : neurons) {
      n.value = {0.0, 0.0, 0.0, 0.0};
      for (size_t i = 0; i < previousLayer->neurons.size(); i++) {
        n.value += previousLayer->neurons.at(i).value * n.weights.at(i);
      }

      // Use activation function
      n.value = n.activationFunction(n.value);
    }
  }

  void backwardPropagation() override {
    // No need to implement for output layer
  }

  void updateWeights(float learningRate) override {
    if (previousLayer == nullptr) {
      return;
    }
    for (Neuron &n : neurons) {
      // Update weights based on neurons in the previous layer
      for (size_t j = 0; j < n.weights.size(); ++j) {
        auto dE_dw = previousLayer->neurons[j].value * n.error;
        dE_dw.clamp();
        n.weights[j] -= learningRate * dE_dw;
      }
      // Update weights based on neighboring neurons
      for (Connection &connection : n.neighbors) {
        auto dE_dw = connection.neuron->value * n.error;
        dE_dw.clamp();
        connection.weight -= learningRate * dE_dw;
      }
    }
  }

  void computeErrors(std::vector<RGBA> const &expectedValues) {
    if (expectedValues.size() != neurons.size()) {
      throw std::invalid_argument("Invalid expected values size");
    }
    size_t i = 0;
    for (auto &neuron : neurons) {
      // Compute the weighted sum of neighboring neuron values
      RGBA neighborSum = {0.0, 0.0, 0.0, 0.0};
      for (auto &connection : neuron.neighbors) {
        neighborSum += connection.weight * connection.neuron->value;
      }

      float weightFactor = 0.5; // Experiment with weight between 0 and 1
      neuron.error = weightFactor * (neuron.value - expectedValues[i]) +
                     (1.0f - weightFactor) * neighborSum;
      ++i;
    }
  }

  std::vector<RGBA> getOutputValues() {
    auto neuronOutput = [](const Neuron &n) { return n.value; };
    auto neuronOutputs = neurons | std::views::transform(neuronOutput);
    const auto &ret =
        std::vector<RGBA>{neuronOutputs.begin(), neuronOutputs.end()};
    return ret;
  }
};
} // namespace sipai