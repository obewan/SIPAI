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
class Network {
public:
  Network() = default;
  ~Network() {
    for (auto layer : layers) {
      delete layer;
    }
  }

  std::vector<Layer *> layers;
  NetworkParameters params;

  /**
   * @brief Performs forward propagation on the network using the given input
   * values.
   *
   * @param inputValues The input values for forward propagation.
   * @return A vector of output values from the output layer after forward
   * propagation.
   */
  std::vector<RGBA> forwardPropagation(const std::vector<RGBA> &inputValues) {
    if (layers.front()->layerType != LayerType::InputLayer) {
      throw NetworkException("Invalid front layer type");
    }
    if (layers.back()->layerType != LayerType::OutputLayer) {
      throw NetworkException("Invalid back layer type");
    }
    ((InputLayer *)layers.front())->setInputValues(inputValues);
    for (auto &layer : layers) {
      layer->forwardPropagation();
    }
    return ((OutputLayer *)layers.back())->getOutputValues();
  }

  /**
   * @brief Performs backward propagation on the network using the given
   * expected values.
   *
   * @param expectedValues The expected values for backward propagation.
   */
  void backwardPropagation(const std::vector<RGBA> &expectedValues) {
    if (layers.back()->layerType != LayerType::OutputLayer) {
      throw NetworkException("Invalid back layer type");
    }
    ((OutputLayer *)layers.back())->computeErrors(expectedValues);
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
      (*it)->backwardPropagation();
    }
  }

  void bindLayers() {
    for (size_t i = 0; i < layers.size(); ++i) {
      if (i > 0) {
        layers.at(i)->previousLayer = layers.at(i - 1);
      }
      if (i < layers.size() - 1) {
        layers.at(i)->nextLayer = layers.at(i + 1);
      }
    }
  }

  void initializeWeights() const {
    for (auto layer : layers) {
      if (layer->previousLayer != nullptr) {
        for (auto &n : layer->neurons) {
          n.initWeights(layer->previousLayer->neurons.size());
        }
      }
    }
  }

  void initializeLayers() {
    auto inputLayer = new InputLayer();
    inputLayer->neurons.resize(params.input_size);
    layers.push_back(inputLayer);

    for (size_t i = 0; i < params.hiddens_count; ++i) {
      auto hiddenLayer = new HiddenLayer();
      hiddenLayer->neurons.resize(params.hidden_size);
      SetActivationFunction(hiddenLayer, params.hidden_activation_function,
                            params.hidden_activation_alpha);
      layers.push_back(hiddenLayer);
    }

    auto outputLayer = new OutputLayer();
    outputLayer->neurons.resize(params.output_size);
    SetActivationFunction(outputLayer, params.output_activation_function,
                          params.output_activation_alpha);
    layers.push_back(outputLayer);

    bindLayers();
    initializeWeights();
  }

  void SetActivationFunction(Layer *layer,
                             EActivationFunction activation_function,
                             float activation_alpha) const {
    switch (activation_function) {
    case EActivationFunction::ELU:
      layer->setActivationFunction(
          [activation_alpha](auto x) { return elu(x, activation_alpha); },
          [activation_alpha](auto x) {
            return eluDerivative(x, activation_alpha);
          });
      break;
    case EActivationFunction::LReLU:
      layer->setActivationFunction(leakyRelu, leakyReluDerivative);
      break;
    case EActivationFunction::PReLU:
      layer->setActivationFunction(
          [activation_alpha](auto x) {
            return parametricRelu(x, activation_alpha);
          },
          [activation_alpha](auto x) {
            return parametricReluDerivative(x, activation_alpha);
          });
      break;
    case EActivationFunction::ReLU:
      layer->setActivationFunction(relu, reluDerivative);
      break;
    case EActivationFunction::Sigmoid:
      layer->setActivationFunction(sigmoid, sigmoidDerivative);
      break;
    case EActivationFunction::Tanh:
      layer->setActivationFunction(tanhFunc, tanhDerivative);
      break;
    default:
      throw NetworkException("Unimplemented Activation Function");
    }
  }
};

} // namespace sipai