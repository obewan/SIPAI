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

  // Create the errors matrix if not already allocated
  if (errors.empty()) {
    errors.create((int)size_y, (int)size_x, CV_32FC4);
  }

  // Iterate over all neurons in the layer
  for (int y = 0; y < (int)size_y; ++y) {
    for (int x = 0; x < (int)size_x; ++x) {
      const Neuron &neuron = neurons[y][x];
      cv::Vec4f &error = errors.at<cv::Vec4f>(y, x);

      // Compute the weighted sum of neighboring neuron values
      cv::Vec4f neighborSum = cv::Vec4f::all(0.0f);
      for (const NeuronConnection &connection : neuron.neighbors) {
        neighborSum += connection.weight.mul(values.at<cv::Vec4f>(
            (int)connection.neuron->index_y, (int)connection.neuron->index_x));
      }

      // Compute and update the error
      const cv::Vec4f &currentValue = values.at<cv::Vec4f>(y, x);
      const cv::Vec4f &expectedValue = expectedValues.at<cv::Vec4f>(y, x);
      const cv::Vec4f newError = weightFactor * (currentValue - expectedValue) +
                                 (1.0f - weightFactor) * neighborSum;
      error = Common::clamp4f(newError, error_min, error_max);
    }
  }
}