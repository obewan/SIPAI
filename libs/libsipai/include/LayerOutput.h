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

  void computeErrors(cv::Mat const &expectedValues);

  cv::Mat getOutputValues() const { return values; }
};
} // namespace sipai