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
      delete layer;
    }
  }

  /**
   * @brief A vector of pointers to Layer objects, representing the layers in
   * the network.
   */
  std::vector<Layer *> layers;

  /**
   * @brief Initializes the layers of the network.
   */
  void initialize();

  /**
   * @brief Performs forward propagation on the network using the given input
   * values.
   *
   * @param inputValues The input values for forward propagation.
   * @return A vector of output values from the output layer after forward
   * propagation.
   */
  std::vector<RGBA> forwardPropagation(const std::vector<RGBA> &inputValues,
                                       bool enableParallax = false);

  /**
   * @brief Performs backward propagation on the network using the given
   * expected values.
   *
   * @param expectedValues The expected values for backward propagation.
   */
  void backwardPropagation(const std::vector<RGBA> &expectedValues,
                           bool enableParallax = false);

  /**
   * @brief Updates the weights of the neurons in the network using the learning
   * rate.
   */
  void updateWeights(float learning_rate, bool enableParallax = false);

  /**
   * @brief Add the neurons layers of the network.
   *
   */
  void addLayers();

  /**
   * @brief Binds the layers of the network together.
   */
  void bindLayers();

  /**
   * @brief Initializes the weights of the neurons in the network.
   */
  void initializeWeights() const;

  /**
   * @brief Initializes the neighbors of the neurons in the network.
   */
  void initializeNeighbors();

  /**
   * @brief Add and initialize the neighbors of a specific neuron
   *
   * @param neuron
   * @param neuron_layer
   * @param neuron_index
   * @param layer_size_x
   * @param layer_size_y
   * @param randomize_weight
   */
  void addNeuronNeighbors(Neuron &neuron, Layer *neuron_layer,
                          size_t neuron_index, int layer_size_x,
                          int layer_size_y, bool randomize_weight = true);

  /**
   * @brief Sets the activation function for a given layer in the network.
   *
   * @param layer A pointer to the Layer object for which the activation
   * function is to be set.
   * @param activation_function An enum representing the type of activation
   * function to be used.
   * @param activation_alpha A float representing the alpha parameter of the
   * activation function.
   */
  void SetActivationFunction(Layer *layer,
                             EActivationFunction activation_function,
                             float activation_alpha) const;

  /**
   * @brief Check if the neural network is initialized.
   *
   * @return true if initialized.
   * @return false
   */
  bool isInitizalized() const { return isInitialized_.load(); }

private:
  std::atomic<bool> isInitialized_ = false;
};

} // namespace sipai