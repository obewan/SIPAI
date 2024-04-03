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
#include <cmath>
#include <cstddef>
#include <execution>
#include <stdexcept>

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
};
} // namespace sipai