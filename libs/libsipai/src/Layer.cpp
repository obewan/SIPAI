#include "Layer.h"
#include "VulkanController.h"
#include <algorithm>
#include <cmath>
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

  if (enable_parallel) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    cv::setNumThreads(num_threads);
  }

  for (size_t row = 0; row < neurons.size(); ++row) {
    for (size_t col = 0; col < neurons[row].size(); ++col) {
      Neuron &currentNeuron = neurons[row][col];
      // Compute matrix multiplication between previous layer values
      // and current neuron weights
      cv::Mat dotProduct = previousLayer->values.mul(currentNeuron.weights);
      // Convert the result matrix to a single value by summing all elements
      float result = cv::sum(dotProduct)[0];
      // Update the neuron value using the activation function
      values.at<cv::Vec4f>(row, col) = activationFunction(result);
    }
  }
}

void Layer::backwardPropagation(const float &error_min, const float &error_max,
                                bool enable_parallel) {
  if (nextLayer == nullptr) {
    return;
  }

  if (enable_parallel) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    cv::setNumThreads(num_threads);
  }

  for (size_t row = 0; row < neurons.size(); ++row) {
    for (size_t col = 0; col < neurons[row].size(); ++col) {
      Neuron &currentNeuron = neurons[row][col];
      cv::Vec4f error(0.0f);
      // Add next layer neurons error ponderated with weights for this neuron
      for (auto &nextLayerNeuronRow : nextLayer->neurons) {
        for (auto &nextLayerNeuron : nextLayerNeuronRow) {
          error += nextLayer->errors
                       .at<cv::Vec4f>(nextLayerNeuron.index_x,
                                      nextLayerNeuron.index_y)
                       .mul(nextLayerNeuron.weights.at<cv::Vec4f>(
                           currentNeuron.index_x, currentNeuron.index_y));
        }
      }
      // Consider errors of adjacent neurons
      for (NeuronConnection &conn : currentNeuron.neighbors) {
        error += conn.weight.mul(
            errors.at<cv::Vec4f>(conn.neuron->index_x, conn.neuron->index_y));
      }
      // Use the derivative of the activation function
      errors.at<cv::Vec4f>(row, col) = sipai::clamp4f(
          activationFunctionDerivative(values.at<cv::Vec4f>(row, col))
              .mul(error),
          error_min, error_max);
    }
  }
}

void Layer::updateWeights(float learningRate, bool enable_parallel) {
  if (previousLayer == nullptr) {
    return;
  }

  if (enable_parallel) {
    unsigned int num_threads = std::thread::hardware_concurrency();
    cv::setNumThreads(num_threads);
  }

  cv::Mat learningRateErrorMat = cv::Mat::zeros(previousLayer->values.size(),
                                                previousLayer->values.type());

  for (size_t row = 0; row < neurons.size(); ++row) {
    for (size_t col = 0; col < neurons[row].size(); ++col) {
      Neuron &neuron = neurons[row][col];

      // Compute learningRateError before the loop
      const cv::Vec4f learningRateError =
          learningRate * errors.at<cv::Vec4f>(neuron.index_x, neuron.index_y);

      // Fill learningRateErrorMat
      for (int y = 0; y < learningRateErrorMat.rows; ++y) {
        for (int x = 0; x < learningRateErrorMat.cols; ++x) {
          learningRateErrorMat.at<cv::Vec4f>(y, x) = learningRateError;
        }
      }

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