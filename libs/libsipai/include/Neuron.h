/**
 * @file Neuron.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Neuron class
 * @date 2024-03-07
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "ActivationFunctions.h"
#include "NeuronConnection.h"
#include "RGBA.h"
#include <exception>
#include <functional>
#include <math.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace sipai {

/**
 * @brief The Neuron class represents a neuron in a neural network. It contains
 * a value, bias, error, and a vector of weights. It also has methods for
 * initializing weights.
 */
class Neuron {
public:
  // Default constructor
  Neuron() = default;

  // The weights of the neuron
  std::vector<RGBA> weights;

  // The value of the neuron
  RGBA value = {0.0, 0.0, 0.0, 0.0};

  // The bias of the neuron
  RGBA bias = {0.0, 0.0, 0.0, 0.0};

  // The error of the neuron
  RGBA error = {0.0, 0.0, 0.0, 0.0};

  // Connections to the adjacents neurons in the same layer, using
  // 4-neighborhood (Von Neumann neighborhood). Could be improve to
  // 8-neighborhood (Moore neighborhood) or Extended neighborhood (radius)
  // later.
  std::vector<NeuronConnection> neighbors;

  /**
   * @brief Initializes the weights of the neuron to a given size. The weights
   * are randomized to break symmetry.
   *
   * @param new_size The new size of the weights vector.
   */
  void initWeights(size_t new_size) {
    weights.resize(new_size);

    // Random initialization
    const float fanIn_fanOut = new_size + 4.0f; // 4 as the 4 values of RGBA
    for (auto &w : weights) {
      w = w.random(fanIn_fanOut);
    }
  }

  std::function<RGBA(RGBA)> activationFunction;
  std::function<RGBA(RGBA)> activationFunctionDerivative;
};
} // namespace sipai