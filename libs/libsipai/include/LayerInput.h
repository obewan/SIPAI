/**
 * @file LayerInput.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Input layer
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "Layer.h"
#include <cstddef>
#include <stdexcept>

namespace sipai {
/**
 * @brief The InputLayer class represents the input layer of a neural network.
 * It inherits from the Layer class and overrides its methods as necessary. This
 * layer is responsible for receiving input from external sources.
 */
class LayerInput : public Layer {
public:
  LayerInput() : Layer(LayerType::LayerInput) {}
  LayerInput(size_t size_x, size_t size_y)
      : Layer(LayerType::LayerInput, size_x, size_y) {}

  void forwardPropagation(bool enableParallax = false) override {
    // No need to implement for input layer
  }

  void backwardPropagation(bool enableParallax = false) override {
    // No need to implement for input layer (no weights of input layer)
  }

  void updateWeights(float learningRate, bool enableParallax = false) override {
    // No need to implement for input layer (no weights of input layer)
  }

  void setInputValues(const std::vector<RGBA> &inputValues) {
    if (inputValues.size() != neurons.size()) {
      throw std::invalid_argument("Invalid input values size");
    }
    for (size_t i = 0; i < neurons.size(); i++) {
      neurons.at(i).value = inputValues.at(i);
    }
  }
};
} // namespace sipai