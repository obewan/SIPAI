#include "NeuralNetwork.h"
#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "SimpleLogger.h"
#include "exception/NeuralNetworkException.h"

using namespace sipai;

image NeuralNetwork::forwardPropagation(const image &inputValues,
                                        bool enable_parallel) {
  if (layers.front()->layerType != LayerType::LayerInput) {
    throw NeuralNetworkException("Invalid front layer type");
  }
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerInput *)layers.front())->setInputValues(inputValues);
  for (auto &layer : layers) {
    layer->forwardPropagation(enable_parallel);
  }
  return ((LayerOutput *)layers.back())->getOutputValues();
}

void NeuralNetwork::backwardPropagation(const image &expectedValues,
                                        bool enable_parallel) {
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerOutput *)layers.back())->computeErrors(expectedValues);
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    (*it)->backwardPropagation(enable_parallel);
  }
}

void NeuralNetwork::updateWeights(float learning_rate, bool enable_parallel) {
  for (auto &layer : layers) {
    layer->updateWeights(learning_rate, enable_parallel);
  }
}
