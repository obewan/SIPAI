#include "NeuralNetwork.h"
#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "SimpleLogger.h"
#include "exception/NeuralNetworkException.h"

using namespace sipai;

std::vector<RGBA>
NeuralNetwork::forwardPropagation(const std::vector<RGBA> &inputValues,
                                  bool enableParallax) {
  if (layers.front()->layerType != LayerType::LayerInput) {
    throw NeuralNetworkException("Invalid front layer type");
  }
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerInput *)layers.front())->setInputValues(inputValues);
  for (auto &layer : layers) {
    layer->forwardPropagation(enableParallax);
  }
  return ((LayerOutput *)layers.back())->getOutputValues();
}

void NeuralNetwork::backwardPropagation(const std::vector<RGBA> &expectedValues,
                                        bool enableParallax) {
  if (layers.back()->layerType != LayerType::LayerOutput) {
    throw NeuralNetworkException("Invalid back layer type");
  }
  ((LayerOutput *)layers.back())->computeErrors(expectedValues);
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    (*it)->backwardPropagation(enableParallax);
  }
}

void NeuralNetwork::updateWeights(float learning_rate, bool enableParallax) {
  for (auto &layer : layers) {
    layer->updateWeights(learning_rate, enableParallax);
  }
}
