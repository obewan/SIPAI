/**
 * @file NeuralNetworkBuilder.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief NeuralNetwork Builder
 * @date 2024-03-30
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "AppParams.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportFacade.h"
#include "NeuralNetworkParams.h"
#include <memory.h>
#include <memory>

namespace sipai {
class NeuralNetworkBuilder {
public:
  NeuralNetworkBuilder();
  NeuralNetworkBuilder(AppParams &appParams,
                       NeuralNetworkParams &networkParams);
  /**
   * @brief Using an external network to build up.
   *
   * @param network
   * @return NeuralNetworkBuilder&
   */
  NeuralNetworkBuilder &with(std::unique_ptr<NeuralNetwork> &network) {
    network_ = std::move(network);
    return *this;
  }

  /**
   * @brief Using external network parameters.
   *
   * @param network_params
   * @return NeuralNetworkBuilder&
   */
  NeuralNetworkBuilder &with(NeuralNetworkParams &network_params) {
    network_params_ = network_params;
    return *this;
  }

  /**
   * @brief Using external app parameters.
   *
   * @param app_params
   * @return NeuralNetworkBuilder&
   */
  NeuralNetworkBuilder &with(const AppParams &app_params) {
    app_params_ = app_params;
    return *this;
  }

  /**
   * @brief Using a progress callback
   *
   * @param progressCallback
   * @return NeuralNetworkBuilder&
   */
  NeuralNetworkBuilder &with(std::function<void(int)> progressCallback) {
    progressCallback_ = progressCallback;
    progressCallbackValue_ = 0;
    return *this;
  }

  /**
   * @brief Create a Or import a neural network
   *
   * @return const NeuralNetworkBuilder&
   */
  NeuralNetworkBuilder &createOrImport();

  /**
   * @brief Add the neurons layers of the network.
   *
   */
  NeuralNetworkBuilder &addLayers();

  /**
   * @brief Add the neighbors connections in a same layer, for all the layers.
   */
  NeuralNetworkBuilder &addNeighbors();

  /**
   * @brief Binds the layers of the network together.
   */
  NeuralNetworkBuilder &bindLayers();

  /**
   * @brief Initializes the weights of the neurons.
   */
  NeuralNetworkBuilder &initializeWeights();

  /**
   * @brief Sets the activation function for the layers.
   *
   */
  NeuralNetworkBuilder &setActivationFunction();

  /**
   * @brief Build the neural network following the methods chain.
   *
   * @return std::unique_ptr<NeuralNetwork>
   */
  std::unique_ptr<NeuralNetwork> build();

private:
  std::unique_ptr<NeuralNetwork> network_ = nullptr;
  AppParams &app_params_;
  NeuralNetworkParams &network_params_;
  bool isImported = false;
  std::function<void(int)> progressCallback_ = {};
  int progressCallbackValue_ = 0;

  void _incrementProgress(int increment) {
    if (progressCallback_) {
      progressCallbackValue_ = progressCallbackValue_ + increment > 100
                                   ? 100
                                   : progressCallbackValue_ + increment;
      progressCallback_(progressCallbackValue_);
    }
  }
};
} // namespace sipai
