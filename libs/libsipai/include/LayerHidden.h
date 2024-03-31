/**
 * @file LayerHidden.h
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
class LayerHidden : public Layer {
public:
  LayerHidden() : Layer(LayerType::LayerHidden) {}

  void forwardPropagation() override {
    if (previousLayer == nullptr) {
      return;
    }
    // Implement forward propagation for hidden layer
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
    if (nextLayer == nullptr) {
      return;
    }

    // Implement backward propagation for hidden layer
    for (size_t i = 0; i < neurons.size(); ++i) {
      neurons[i].error = {0.0, 0.0, 0.0, 0.0};
      for (Neuron &n : nextLayer->neurons) {
        neurons[i].error += n.weights[i] * n.error;
      }
      // Consider errors of adjacent neurons
      for (NeuronConnection &connection : neurons[i].neighbors) {
        neurons[i].error += connection.neuron->error * connection.weight;
      }

      // Use the derivative of the activation function
      neurons[i].error *=
          neurons[i].activationFunctionDerivative(neurons[i].value);
    }
  }

  void updateWeights(float learningRate) override {
    if (previousLayer == nullptr) {
      return;
    }

    for (Neuron &n : neurons) {
      for (size_t j = 0; j < n.weights.size(); ++j) {
        auto dE_dw = previousLayer->neurons[j].value * n.error;
        dE_dw.clamp();
        n.weights[j] -= learningRate * dE_dw;
      }
      // Update weights based on neighboring neurons
      for (NeuronConnection &connection : n.neighbors) {
        auto dE_dw = connection.neuron->value * n.error;
        dE_dw.clamp();
        connection.weight -= learningRate * dE_dw;
      }
    }
  }
};
} // namespace sipai