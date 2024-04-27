#include "LayerOutput.h"
#include "Manager.h"
#include <opencv2/core/matx.hpp>

using namespace sipai;

void LayerOutput::computeErrors(const cv::Mat &expectedValues) {
  if (expectedValues.total() != total()) {
    throw std::invalid_argument("Invalid expected values size");
  }

  const float error_min = Manager::getConstInstance().network_params.error_min;
  const float error_max = Manager::getConstInstance().network_params.error_max;
  const float weightFactor = 0.5f; // Experiment with weight between 0 and 1

  const int rows = neurons.size();
  const int cols = neurons[0].size();

  // Create the errors matrix if not already allocated
  if (errors.empty()) {
    errors.create(rows, cols, CV_32FC4);
  }

  // Iterate over all neurons in the layer
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const Neuron &neuron = neurons[i][j];
      cv::Vec4f &error = errors.at<cv::Vec4f>(i, j);

      // Compute the weighted sum of neighboring neuron values
      cv::Vec4f neighborSum = cv::Vec4f(0.0f);
      for (const NeuronConnection &connection : neuron.neighbors) {
        neighborSum += connection.weight.mul(values.at<cv::Vec4f>(
            connection.neuron->index_x, connection.neuron->index_y));
      }

      // Compute the error
      const cv::Vec4f &currentValue = values.at<cv::Vec4f>(i, j);
      const cv::Vec4f &expectedValue = expectedValues.at<cv::Vec4f>(i, j);
      error = sipai::clamp4f(weightFactor * (currentValue - expectedValue) +
                                 (1.0f - weightFactor) * neighborSum,
                             error_min, error_max);
    }
  }
}