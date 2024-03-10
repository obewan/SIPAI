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
#include "Common.h"
#include <functional>
#include <math.h>
#include <random>
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
  std::vector<Neuron *> neighbors;

  /**
   * @brief Initializes the weights of the neuron to a given size. The weights
   * are randomized to break symmetry.
   *
   * @param new_size The new size of the weights vector.
   */
  void initWeights(size_t new_size) {
    weights.resize(new_size);

    // randomize weights to break symmetry
    std::random_device rd;
    for (auto &w : weights) {
      std::mt19937 gen(rd());
      std::normal_distribution<float> dist(0.1f, 0.01f);
      std::for_each(w.begin(), w.end(), [&gen, &dist](float &f) {
        float distg = dist(gen);
        f = std::max(0.0f, std::min(0.1f, distg));
      });
    }
  }

  std::function<float(float)> activationFunction;
  std::function<float(float)> activationFunctionDerivative;
};
} // namespace sipai