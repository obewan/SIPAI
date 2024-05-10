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
#include "Common.h"
#include "Layer.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

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
   * @return A vector of output values from the output layer after forward
   * propagation.
   */
  cv::Mat forwardPropagation(const cv::Mat &inputValues);

  /**
   * @brief Performs backward propagation on the network using the given
   * expected values.
   *
   * @param expectedValues The expected values for backward propagation.
   * @param error_min error minimum
   * @param error_max error maximum
   */
  void backwardPropagation(const cv::Mat &expectedValues,
                           const float &error_min, const float &error_max);

  /**
   * @brief Updates the weights of the neurons in the network using the learning
   * rate.
   *
   * @param learning_rate The learning rate
   */
  void updateWeights(float learning_rate);

  /**
   * @brief max weights of all neurons, useful for csv export
   * It is also the maximum layer neurons.
   * Updated during neural network import or creation
   */
  size_t max_weights = 0;
};

} // namespace sipai