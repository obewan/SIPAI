#include "NeuralNetwork.h"
#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "SimpleLogger.h"
#include "exception/NeuralNetworkException.h"

using namespace sipai;

cv::Mat NeuralNetwork::forwardPropagation(const cv::Mat &inputValues,
                                          bool enable_vulkan) {
  if (layers.front()->layerType != LayerType::LayerInput) {
    throw NeuralNetworkException("Invalid front layer type");
  }
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerInput *)layers.front())->setInputValues(inputValues);
  for (auto &layer : layers) {
    layer->forwardPropagation(enable_vulkan);
  }
  return ((LayerOutput *)layers.back())->getOutputValues();
}

void NeuralNetwork::backwardPropagation(const cv::Mat &expectedValues,
                                        const float &error_min,
                                        const float &error_max) {
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerOutput *)layers.back())->computeErrors(expectedValues);
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    (*it)->backwardPropagation(error_min, error_max);
  }
}

void NeuralNetwork::updateWeights(float learning_rate) {
  for (auto &layer : layers) {
    layer->updateWeights(learning_rate);
  }
}
