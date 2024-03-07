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
        auto const &prev_n = previousLayer->neurons.at(i);
        size_t irgba = 0;
        std::for_each(n.value.begin(), n.value.end(),
                      [&irgba, &n, &i, &prev_n](float &value) {
                        value +=
                            prev_n.value.at(irgba) * n.weights.at(i).at(irgba);
                        ++irgba;
                      });
      }
      // Use activation function
      std::for_each(n.value.begin(), n.value.end(), [&n](float &value) {
        value = n.activationFunction(value);
      });
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
      for (size_t j = 0; j < n.weights.size(); ++j) {
        for (size_t j = 0; j < n.weights.size(); ++j) {
          size_t irgba = 0;
          std::for_each(n.weights[j].begin(), n.weights[j].end(),
                        [this, &j, &n, &irgba, &learningRate]() {
                          // Gradient descent
                          float dE_dw =
                              previousLayer->neurons[j].value.at(irgba) *
                              n.error.at(irgba);
                          // Update weights
                          n.weights[j].at(irgba) -= learningRate * dE_dw;
                        });
        }
      }
    }
  }

  void computeErrors(std::vector<RGBA> const &expectedValues) {
    if (expectedValues.size() != neurons.size()) {
      throw std::invalid_argument("Invalid expected values size");
    }
    size_t i = 0;
    for (auto &neuron : neurons) {
      size_t irgba = 0;
      std::for_each(neuron.error.begin(), neuron.error.end(),
                    [this, &irgba, &i, &neuron, &expectedValues](float &error) {
                      error =
                          neuron.value.at(irgba) - expectedValues[i].at(irgba);
                      ++irgba;
                    });
      ++i;
    }
  }

  std::vector<RGBA> getOutputValues() {
    auto neuronOutput = [](const Neuron &n) { return n.value; };
    auto neuronOutputs = neurons | std::views::transform(neuronOutput);
    return std::vector<RGBA>{neuronOutputs.begin(), neuronOutputs.end()};
  }
};
} // namespace sipai