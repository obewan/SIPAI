/**
 * @file HiddenLayer.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Hidden layer
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "Layer.h"
#include <cstddef>

namespace sipai {
/**
 * @brief The HiddenLayer class represents a hidden layer in a neural network.
 * It inherits from the Layer class and overrides its methods as necessary.
 * Hidden layers are responsible for processing inputs received from the input
 * layer and passing the result to the output layer or the next hidden layer.
 */
class HiddenLayer : public Layer {
public:
  HiddenLayer() : Layer(LayerType::HiddenLayer) {}

  void forwardPropagation() override {
    if (previousLayer == nullptr) {
      return;
    }
    // Implement forward propagation for hidden layer
    for (auto &n : neurons) {
      n.value = {0.0, 0.0, 0.0, 0.0};
      for (size_t i = 0; i < previousLayer->neurons.size(); i++) {
        auto const &prev_n = previousLayer->neurons.at(i);
        size_t irgba = 0;
        std::for_each(n.value.begin(), n.value.end(),
                      [&n, &i, &prev_n, &irgba](float &value) {
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
    if (nextLayer == nullptr) {
      return;
    }

    // Implement backward propagation for hidden layer
    for (size_t i = 0; i < neurons.size(); ++i) {
      neurons[i].error = {0.0, 0.0, 0.0, 0.0};
      for (Neuron &n : nextLayer->neurons) {
        size_t index = 0;
        std::for_each(neurons[i].error.begin(), neurons[i].error.end(),
                      [&index, &i, &n](float &error) {
                        error += n.weights[i].at(index) * n.error.at(index);
                        ++index;
                      });
      }
      // Use the derivative of the activation function
      size_t index2 = 0;
      std::for_each(neurons[i].error.begin(), neurons[i].error.end(),
                    [this, &i, &index2](float &error) {
                      error *= neurons[i].activationFunctionDerivative(
                          neurons[i].value.at(index2));
                      ++index2;
                    });
    }
  }

  void updateWeights(float learningRate) override {
    if (previousLayer == nullptr) {
      return;
    }
    for (Neuron &n : neurons) {
      for (size_t j = 0; j < n.weights.size(); ++j) {
        size_t index = 0;
        std::for_each(n.weights[j].begin(), n.weights[j].end(),
                      [this, &j, &n, &index, &learningRate]() {
                        // Gradient descent
                        float dE_dw =
                            previousLayer->neurons[j].value.at(index) *
                            n.error.at(index);
                        // Update weights
                        n.weights[j].at(index) -= learningRate * dE_dw;
                      });
      }
    }
  }
};
} // namespace sipai