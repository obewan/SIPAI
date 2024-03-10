/**
 * @file Network.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Network
 * @date 2024-03-08
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "HiddenLayer.h"
#include "InputLayer.h"
#include "Layer.h"
#include "NetworkParameters.h"
#include "OutputLayer.h"
#include "exception/NetworkException.h"

namespace sipai {

/**
 * @class Network
 * @brief This class represents a neural network for image processing.
 */
class Network {
public:
  /**
   * @brief Default constructor for the Network class.
   */
  Network() = default;

  /**
   * @brief Destructor for the Network class. It deletes all layers in the
   * network.
   */
  ~Network() {
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
   * @brief Performs forward propagation on the network using the given input
   * values.
   *
   * @param inputValues The input values for forward propagation.
   * @return A vector of output values from the output layer after forward
   * propagation.
   */
  std::vector<RGBA> forwardPropagation(const std::vector<RGBA> &inputValues);

  /**
   * @brief Performs backward propagation on the network using the given
   * expected values.
   *
   * @param expectedValues The expected values for backward propagation.
   */
  void backwardPropagation(const std::vector<RGBA> &expectedValues);

  /**
   * @brief Binds the layers of the network together.
   */
  void bindLayers();

  /**
   * @brief Initializes the weights of the neurons in the network.
   */
  void initializeWeights() const;

  /**
   * @brief Initializes the layers of the network.
   */
  void initializeLayers();

  /**
   * @brief Initializes the neighbors of the neurons in the network.
   */
  void initializeNeighbors();

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
};

} // namespace sipai