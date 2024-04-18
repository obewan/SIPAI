/**
 * @file LayerOutput.h
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
#include <execution>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace sipai {
/**
 * @brief The OutputLayer class represents the output layer of a neural network.
 * It inherits from the Layer class and overrides its methods as necessary. This
 * layer is responsible for producing the final output of the network.
 */
class LayerOutput : public Layer {
public:
  LayerOutput() : Layer(LayerType::LayerOutput) {}
  LayerOutput(size_t size_x, size_t size_y)
      : Layer(LayerType::LayerOutput, size_x, size_y) {}

  void computeErrors(std::vector<RGBA> const &expectedValues) {
    if (expectedValues.size() != neurons.size()) {
      throw std::invalid_argument("Invalid expected values size");
    }

    float error_min = -1.0f;
    float error_max = 1.0f;
    float weightFactor = 0.5; // Experiment with weight between 0 and 1

    for (auto &neuron : neurons) {
      // Compute the weighted sum of neighboring neuron values
      RGBA neighborSum = {0.0, 0.0, 0.0, 0.0};
      for (auto &connection : neuron.neighbors) {
        neighborSum += connection.weight * connection.neuron->value;
      }
      size_t pos = &neuron - &neurons[0];
      neuron.error = (weightFactor * (neuron.value - expectedValues[pos]) +
                      (1.0f - weightFactor) * neighborSum)
                         .clamp(error_min, error_max);
    }
  }

  std::vector<RGBA> getOutputValues() {
    auto neuronOutputs =
        std::views::transform(neurons, [](const Neuron &n) { return n.value; });
    return std::vector<RGBA>(neuronOutputs.begin(), neuronOutputs.end());
  }
};
} // namespace sipai