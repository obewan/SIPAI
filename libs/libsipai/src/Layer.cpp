#include "Layer.h"
#include "VulkanController.h"
#include <algorithm>
#include <cmath>
#include <opencv2/core/hal/interface.h>
#include <stdexcept>

using namespace sipai;

void Layer::forwardPropagation(bool enable_vulkan, bool enable_parallel) {
  if (previousLayer == nullptr) {
    return;
  }

  if (enable_vulkan) {
    VulkanController::getInstance().forwardPropagation(previousLayer, this);
    return;
  }

  cv::setNumThreads(enable_parallel ? std::thread::hardware_concurrency() : 0);

  for (size_t y = 0; y < neurons.size(); ++y) {
    for (size_t x = 0; x < neurons[y].size(); ++x) {
      Neuron &currentNeuron = neurons[y][x];
      // Compute matrix multiplication between previous layer values
      // and current neuron weights
      cv::Mat dotProduct = previousLayer->values.mul(currentNeuron.weights);
      // Convert the result matrix to a single value by summing all elements
      float result = cv::sum(dotProduct)[0];
      // Update the neuron value using the activation function
      values.at<cv::Vec4f>(x, y) = activationFunction(result);
    }
  }
}

void Layer::backwardPropagation(const float &error_min, const float &error_max,
                                bool enable_parallel) {
  if (nextLayer == nullptr) {
    return;
  }

  cv::setNumThreads(enable_parallel ? std::thread::hardware_concurrency() : 0);

  for (size_t y = 0; y < neurons.size(); ++y) {
    for (size_t x = 0; x < neurons[y].size(); ++x) {
      Neuron &currentNeuron = neurons[y][x];
      cv::Vec4f error(0.0f);
      const cv::Mat nextLayerErrors = nextLayer->errors;

      // Add next layer neurons error ponderated with weights for this neuron
      for (const auto &nextLayerNeuronRow : nextLayer->neurons) {
        for (const auto &nextLayerNeuron : nextLayerNeuronRow) {
          const cv::Vec4f currentError = nextLayerErrors.at<cv::Vec4f>(
              nextLayerNeuron.index_x, nextLayerNeuron.index_y);
          const cv::Vec4f weight = nextLayerNeuron.weights.at<cv::Vec4f>(x, y);
          error += currentError.mul(weight);
        }
      }
      // Consider errors of adjacent neurons
      for (const NeuronConnection &conn : currentNeuron.neighbors) {
        error += conn.weight.mul(
            errors.at<cv::Vec4f>(conn.neuron->index_x, conn.neuron->index_y));
      }
      // Use the derivative of the activation function
      const cv::Vec4f activationDerivative =
          activationFunctionDerivative(values.at<cv::Vec4f>(x, y));
      const cv::Vec4f clampedError =
          sipai::clamp4f(activationDerivative.mul(error), error_min, error_max);

      errors.at<cv::Vec4f>(x, y) = clampedError;
    }
  }
}

void Layer::updateWeights(float learningRate, bool enable_parallel) {
  if (previousLayer == nullptr) {
    return;
  }

  cv::setNumThreads(enable_parallel ? std::thread::hardware_concurrency() : 0);

  for (size_t y = 0; y < neurons.size(); ++y) {
    for (size_t x = 0; x < neurons[y].size(); ++x) {
      Neuron &neuron = neurons[y][x];

      // Get the error of current neuron, mult by the learningRate
      const cv::Vec4f learningRateError =
          errors.at<cv::Vec4f>(x, y) * learningRate;

      // Create a matrix with dimensions of neuron weights
      // and previous learningRateError
      cv::Mat learningRateErrorMat(neuron.weights.size(), neuron.weights.type(),
                                   learningRateError);

      // Update neuron weights that are connections weights with previous layers
      neuron.weights -= previousLayer->values.mul(learningRateErrorMat);

      // Update weights based on neighboring neurons
      for (NeuronConnection &conn : neuron.neighbors) {
        conn.weight -=
            values.at<cv::Vec4f>(conn.neuron->index_x, conn.neuron->index_y)
                .mul(learningRateError);
      }
    }
  }
}