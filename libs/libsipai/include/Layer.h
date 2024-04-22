/**
 * @file Layer.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Abstract layer class
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "Neuron.h"
#include "VulkanController.h"
#include "exception/NeuralNetworkException.h"
#include <atomic>
#include <cstddef>
#include <execution>
#include <functional>
#include <map>
#include <thread>

namespace sipai {
enum class LayerType { LayerInput, LayerHidden, LayerOutput };

const std::map<std::string, LayerType, std::less<>> layer_map{
    {"LayerInput", LayerType::LayerInput},
    {"LayerHidden", LayerType::LayerHidden},
    {"LayerOutput", LayerType::LayerOutput}};

/**
 * @brief The Layer class represents a layer in a neural network. It contains a
 * vector of Neurons and has methods for forward propagation, backward
 * propagation, and updating weights.
 */
class Layer {
public:
  explicit Layer(LayerType layerType, size_t size_x = 0, size_t size_y = 0)
      : layerType(layerType), size_x(size_x), size_y(size_y) {}
  virtual ~Layer() = default;

  const LayerType layerType;
  std::vector<Neuron> neurons;
  Layer *previousLayer = nullptr;
  Layer *nextLayer = nullptr;
  size_t size_x = 0;
  size_t size_y = 0;

  const std::string UndefinedLayer = "UndefinedLayer";

  /**
   * @brief Performs forward propagation using the previous layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void forwardPropagation(bool enable_vulkan = false,
                                  bool enable_parallel = false) {
    if (previousLayer == nullptr) {
      return;
    }

    auto updateNeuron = [](Neuron &n, Layer *previousLayer) {
      n.value = {0.0, 0.0, 0.0, 0.0};
      for (size_t i = 0; i < previousLayer->neurons.size(); i++) {
        n.value += previousLayer->neurons[i].value * n.weights[i];
      }
      // Use activation function
      n.value = n.activationFunction(n.value);
    };
    if (enable_vulkan) {
      auto &vulkanController = VulkanController::getInstance();
      if (!vulkanController.IsInitialized()) {
        throw NeuralNetworkException("Vulkan controller is not initialized.");
      }
      // Prepare data for the shader
      vulkanController.copyNeuronsDataToBuffer(neurons);
      // Run the shader
      vulkanController.computeShader(vulkanController.forwardShader, neurons);
      // Get the results
      vulkanController.copyBufferToNeuronsData(neurons);

    } else if (enable_parallel) {
      std::vector<std::jthread> threads;
      size_t num_threads =
          std::min((size_t)std::thread::hardware_concurrency(), neurons.size());
      std::atomic<size_t> current_index = 0; // atomicity between threads
      for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([this, &updateNeuron, &current_index]() {
          while (true) {
            size_t index = current_index.fetch_add(1);
            if (index >= neurons.size()) {
              break;
            }
            updateNeuron(neurons[index], previousLayer);
          }
        });
      }
      for (auto &thread : threads) {
        thread.join();
      }

    } else {
      for (auto &n : neurons) {
        updateNeuron(n, previousLayer);
      }
    }
  };

  /**
   * @brief Performs backward propagation using the next layer.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void backwardPropagation(const float &error_min,
                                   const float &error_max,
                                   bool enable_parallel = false) {
    if (nextLayer == nullptr) {
      return;
    }

    auto updateNeuron = [](Neuron &n, const size_t &pos, Layer *nextLayer,
                           const float &error_min, const float &error_max) {
      RGBA error = {0.0, 0.0, 0.0, 0.0};
      for (Neuron &nn : nextLayer->neurons) {
        error += nn.weights[pos] * nn.error;
      }
      // Consider errors of adjacent neurons
      for (NeuronConnection &conn : n.neighbors) {
        error += conn.neuron->error * conn.weight;
      }
      // Use the derivative of the activation function
      n.error = (error * n.activationFunctionDerivative(n.value))
                    .clamp(error_min, error_max);
    };

    if (enable_parallel) {
      std::vector<std::jthread> threads;
      size_t num_threads =
          std::min((size_t)std::thread::hardware_concurrency(), neurons.size());
      std::atomic<size_t> current_index = 0; // atomicity between threads
      for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            [this, &updateNeuron, &current_index, error_min, error_max]() {
              while (true) {
                size_t index = current_index.fetch_add(1);
                if (index >= neurons.size()) {
                  break;
                }
                updateNeuron(neurons[index], index, nextLayer, error_min,
                             error_max);
              }
            });
      }
      for (auto &thread : threads) {
        thread.join();
      }

    } else {
      for (size_t index = 0; index < neurons.size(); ++index) {
        updateNeuron(neurons[index], index, nextLayer, error_min, error_max);
      }
    }
  }

  /**
   * @brief Updates the weights of the neurons in this layer using the
   * previous layer and a learning rate.
   *
   * @param learningRate The learning rate to use when updating weights.
   * @param enable_parallel enable parallelism (experimental)
   */
  virtual void updateWeights(float learningRate, bool enable_parallel = false) {
    if (previousLayer == nullptr) {
      return;
    }

    auto updateNeuron = [](Neuron &n, Layer *previousLayer,
                           const float &learningRate) {
      const auto learningRateError = learningRate * n.error;
      // Update weights based on neurons in the previous layer
      for (size_t k = 0; k < n.weights.size(); ++k) {
        n.weights[k] -= previousLayer->neurons[k].value * learningRateError;
      }
      // Update weights based on neighboring neurons
      for (NeuronConnection &conn : n.neighbors) {
        conn.weight -= conn.neuron->value * learningRateError;
      }
    };

    if (enable_parallel) {
      std::vector<std::jthread> threads;
      size_t num_threads =
          std::min((size_t)std::thread::hardware_concurrency(), neurons.size());
      std::atomic<size_t> current_index = 0; // atomicity between threads
      for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(
            [this, &updateNeuron, &current_index, learningRate]() {
              while (true) {
                size_t index = current_index.fetch_add(1);
                if (index >= neurons.size()) {
                  break;
                }
                updateNeuron(neurons[index], previousLayer, learningRate);
              }
            });
      }
      for (auto &thread : threads) {
        thread.join();
      }

    } else {
      for (auto &n : neurons) {
        updateNeuron(n, previousLayer, learningRate);
      }
    }
  }

  const std::string getLayerTypeStr() const {
    for (const auto &[key, mLayerType] : layer_map) {
      if (mLayerType == layerType) {
        return key;
      }
    }
    return UndefinedLayer;
  }

  void setActivationFunction(const std::function<RGBA(RGBA)> &function,
                             const std::function<RGBA(RGBA)> &derivative) {
    for (auto &n : neurons) {
      n.activationFunction = function;
      n.activationFunctionDerivative = derivative;
    }
  }
};
} // namespace sipai