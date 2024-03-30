#include "NeuralNetwork.h"
#include "HiddenLayer.h"
#include "InputLayer.h"
#include "Manager.h"
#include "OutputLayer.h"
#include "SimpleLogger.h"
#include "exception/NetworkException.h"

using namespace sipai;

void NeuralNetwork::initialize() {
  if (isInitialized_) {
    return;
  }
  SimpleLogger::LOG_INFO("Initializing the neural network...");
  SimpleLogger::LOG_INFO("Adding layers...");
  addLayers();

  SimpleLogger::LOG_INFO("Binding layers...");
  bindLayers();

  SimpleLogger::LOG_INFO("Initializing layers neurons weights...");
  initializeWeights();

  SimpleLogger::LOG_INFO("Initializing layers neurons neighbors...");
  initializeNeighbors();

  SimpleLogger::LOG_INFO("Initializing layers done.");
  isInitialized_ = true;
}

std::vector<RGBA>
NeuralNetwork::forwardPropagation(const std::vector<RGBA> &inputValues) {
  if (layers.front()->layerType != LayerType::InputLayer) {
    throw NetworkException("Invalid front layer type");
  }
  if (layers.back()->layerType != LayerType::OutputLayer) {
    throw NetworkException("Invalid back layer type");
  }
  ((InputLayer *)layers.front())->setInputValues(inputValues);
  for (auto &layer : layers) {
    layer->forwardPropagation();
  }
  return ((OutputLayer *)layers.back())->getOutputValues();
}

void NeuralNetwork::backwardPropagation(
    const std::vector<RGBA> &expectedValues) {
  if (layers.back()->layerType != LayerType::OutputLayer) {
    throw NetworkException("Invalid back layer type");
  }
  ((OutputLayer *)layers.back())->computeErrors(expectedValues);
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    (*it)->backwardPropagation();
  }
}

void NeuralNetwork::addLayers() {
  auto inputLayer = new InputLayer();
  const auto &network_params = Manager::getInstance().network_params;
  inputLayer->neurons.resize(network_params.input_size_x *
                             network_params.input_size_y);
  layers.push_back(inputLayer);

  for (size_t i = 0; i < network_params.hiddens_count; ++i) {
    auto hiddenLayer = new HiddenLayer();
    hiddenLayer->neurons.resize(network_params.hidden_size_x *
                                network_params.hidden_size_y);
    SetActivationFunction(hiddenLayer,
                          network_params.hidden_activation_function,
                          network_params.hidden_activation_alpha);
    layers.push_back(hiddenLayer);
  }

  auto outputLayer = new OutputLayer();
  outputLayer->neurons.resize(network_params.output_size_x *
                              network_params.output_size_y);
  SetActivationFunction(outputLayer, network_params.output_activation_function,
                        network_params.output_activation_alpha);
  layers.push_back(outputLayer);
}

void NeuralNetwork::bindLayers() {
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i > 0) {
      layers.at(i)->previousLayer = layers.at(i - 1);
    }
    if (i < layers.size() - 1) {
      layers.at(i)->nextLayer = layers.at(i + 1);
    }
  }
}

void NeuralNetwork::initializeWeights() const {
  for (auto layer : layers) {
    if (layer->previousLayer != nullptr) {
      for (auto &n : layer->neurons) {
        n.initWeights(layer->previousLayer->neurons.size());
      }
    }
  }
}

void NeuralNetwork::initializeNeighbors() {
  const auto &network_params = Manager::getInstance().network_params;
  for (auto &layer : layers) {
    int layer_size_x = 0;
    int layer_size_y = 0;

    if (dynamic_cast<InputLayer *>(layer)) {
      layer_size_x = network_params.input_size_x;
      layer_size_y = network_params.input_size_y;
    } else if (dynamic_cast<HiddenLayer *>(layer)) {
      layer_size_x = network_params.hidden_size_x;
      layer_size_y = network_params.hidden_size_y;
    } else if (dynamic_cast<OutputLayer *>(layer)) {
      layer_size_x = network_params.output_size_x;
      layer_size_y = network_params.output_size_y;
    } else {
      throw NetworkException("Invalid layer type");
    }

    for (size_t i = 0; i < layer->neurons.size(); ++i) {
      addNeuronNeighbors(layer->neurons[i], layer, i, layer_size_x,
                         layer_size_y);
    }
  }
}

void NeuralNetwork::addNeuronNeighbors(Neuron &neuron, Layer *neuron_layer,
                                       size_t neuron_index, int layer_size_x,
                                       int layer_size_y,
                                       bool randomize_weight) {
  // Compute the coordinates of the neuron in the 2D grid
  int x = neuron_index % layer_size_x;
  int y = neuron_index / layer_size_x;

  // For each possible direction (up, down, left, right), check if there
  // is a neuron in that direction and, if so, establish a connection
  std::vector<std::pair<int, int>> directions = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
  for (auto [dx, dy] : directions) {
    int nx = x + dx;
    int ny = y + dy;
    if (nx >= 0 && nx < layer_size_x && ny >= 0 && ny < layer_size_y) {
      int ni = ny * layer_size_x + nx;
      Neuron &neighbor = neuron_layer->neurons[ni];

      RGBA weight;
      if (randomize_weight) {
        // Create a connection with a random initial weight
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        weight = {dist(gen), dist(gen), dist(gen), dist(gen)};
      }

      neuron.neighbors.push_back(Connection(&neighbor, weight));
    }
  }
}

void NeuralNetwork::SetActivationFunction(
    Layer *layer, EActivationFunction activation_function,
    float activation_alpha) const {
  switch (activation_function) {
  case EActivationFunction::ELU:
    layer->setActivationFunction(
        [activation_alpha](auto x) { return elu(x, activation_alpha); },
        [activation_alpha](auto x) {
          return eluDerivative(x, activation_alpha);
        });
    break;
  case EActivationFunction::LReLU:
    layer->setActivationFunction(leakyRelu, leakyReluDerivative);
    break;
  case EActivationFunction::PReLU:
    layer->setActivationFunction(
        [activation_alpha](auto x) {
          return parametricRelu(x, activation_alpha);
        },
        [activation_alpha](auto x) {
          return parametricReluDerivative(x, activation_alpha);
        });
    break;
  case EActivationFunction::ReLU:
    layer->setActivationFunction(relu, reluDerivative);
    break;
  case EActivationFunction::Sigmoid:
    layer->setActivationFunction(sigmoid, sigmoidDerivative);
    break;
  case EActivationFunction::Tanh:
    layer->setActivationFunction(tanhFunc, tanhDerivative);
    break;
  default:
    throw NetworkException("Unimplemented Activation Function");
  }
}

void NeuralNetwork::updateWeights(float learning_rate) {
  for (auto &layer : layers) {
    layer->updateWeights(learning_rate);
  }
}