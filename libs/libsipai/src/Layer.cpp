#include "Layer.h"
#include "VulkanController.h"

using namespace sipai;

void Layer::forwardPropagation(bool enable_vulkan, bool enable_parallel) {
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
    VulkanController::getInstance().forwardPropagation(previousLayer, this);

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
}

void Layer::backwardPropagation(const float &error_min, const float &error_max,
                                bool enable_parallel) {
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
      threads.emplace_back([this, &updateNeuron, &current_index, error_min,
                            error_max]() {
        while (true) {
          size_t index = current_index.fetch_add(1);
          if (index >= neurons.size()) {
            break;
          }
          updateNeuron(neurons[index], index, nextLayer, error_min, error_max);
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

void Layer::updateWeights(float learningRate, bool enable_parallel) {
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