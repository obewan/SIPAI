#include "LayerOutput.h"
#include "Manager.h"
#include <cstddef>
#include <opencv2/core/matx.hpp>

using namespace sipai;

void LayerOutput::computeErrors(const cv::Mat &expectedValues) {
  if (expectedValues.total() != total()) {
    throw std::invalid_argument("Invalid expected values size");
  }

  const float error_min = Manager::getConstInstance().network_params.error_min;
  const float error_max = Manager::getConstInstance().network_params.error_max;
  const float weightFactor = 0.5f; // Experiment with weight between 0 and 1

  const size_t rows = neurons.size();
  const size_t cols = neurons[0].size();

  // Create the errors matrix if not already allocated
  if (errors.empty()) {
    errors.create((int)cols, (int)rows, CV_32FC4);
  }

  // Iterate over all neurons in the layer
  for (int y = 0; y < (int)rows; ++y) {
    for (int x = 0; x < (int)cols; ++x) {
      const Neuron &neuron = neurons[y][x];
      cv::Vec4f &error = errors.at<cv::Vec4f>(x, y);

      // Compute the weighted sum of neighboring neuron values
      cv::Vec4f neighborSum = cv::Vec4f(0.0f);
      for (const NeuronConnection &connection : neuron.neighbors) {
        neighborSum += connection.weight.mul(values.at<cv::Vec4f>(
            (int)connection.neuron->index_x, (int)connection.neuron->index_y));
      }

      // Compute and update the error
      const cv::Vec4f &currentValue = values.at<cv::Vec4f>(x, y);
      const cv::Vec4f &expectedValue = expectedValues.at<cv::Vec4f>(x, y);
      const cv::Vec4f newError = weightFactor * (currentValue - expectedValue) +
                                 (1.0f - weightFactor) * neighborSum;
      error = sipai::clamp4f(newError, error_min, error_max);
    }
  }
}