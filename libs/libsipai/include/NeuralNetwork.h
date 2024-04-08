/**
 * @file NeuralNetwork.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief NeuralNetwork
 * @date 2024-03-08
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Layer.h"
#include <atomic>
#include <cstddef>

namespace sipai {

/**
 * @class NeuralNetwork
 * @brief This class represents a neural network for image processing.
 */
class NeuralNetwork {
public:
  /**
   * @brief Default constructor for the Network class.
   */
  NeuralNetwork() = default;
  // Rule of Five:
  NeuralNetwork(const NeuralNetwork &other) = delete; // Copy constructor
  NeuralNetwork &
  operator=(const NeuralNetwork &other) = delete; // Copy assignment operator
  NeuralNetwork(NeuralNetwork &&other) = delete;  // Move constructor
  NeuralNetwork &
  operator=(NeuralNetwork &&other) = delete; // Move assignment operator
  ~NeuralNetwork() {
    for (auto layer : layers) {
      if (layer != nullptr) {
        delete layer;
      }
    }
  }

  /**
   * @brief A vector of pointers to Layer objects, representing the layers in
   * the network.
   */
  std::vector<Layer *> layers;

  /**
   * @brief Performs forward propagation on the network using the given input
   * values.
   *
   * @param inputValues The input values for forward propagation.
   * @param enable_parallel enable parallelism (experimental)
   * @return A vector of output values from the output layer after forward
   * propagation.
   */
  std::vector<RGBA> forwardPropagation(const std::vector<RGBA> &inputValues,
                                       bool enable_parallel = false);

  /**
   * @brief Performs backward propagation on the network using the given
   * expected values.
   *
   * @param expectedValues The expected values for backward propagation.
   * @param enable_parallel enable parallelism (experimental)
   */
  void backwardPropagation(const std::vector<RGBA> &expectedValues,
                           bool enable_parallel = false);

  /**
   * @brief Updates the weights of the neurons in the network using the learning
   * rate.
   *
   * @param learning_rate The learning rate
   * @param enable_parallel enable parallelism (experimental)
   */
  void updateWeights(float learning_rate, bool enable_parallel = false);

  /**
   * @brief max weights of all neurons, useful for csv export
   *
   */
  size_t max_weights = 0;
};

} // namespace sipai