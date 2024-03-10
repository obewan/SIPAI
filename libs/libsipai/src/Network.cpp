#include "Network.h"
#include "Manager.h"

using namespace sipai;

std::vector<RGBA>
Network::forwardPropagation(const std::vector<RGBA> &inputValues) {
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

void Network::backwardPropagation(const std::vector<RGBA> &expectedValues) {
  if (layers.back()->layerType != LayerType::OutputLayer) {
    throw NetworkException("Invalid back layer type");
  }
  ((OutputLayer *)layers.back())->computeErrors(expectedValues);
  for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
    (*it)->backwardPropagation();
  }
}

void Network::bindLayers() {
  for (size_t i = 0; i < layers.size(); ++i) {
    if (i > 0) {
      layers.at(i)->previousLayer = layers.at(i - 1);
    }
    if (i < layers.size() - 1) {
      layers.at(i)->nextLayer = layers.at(i + 1);
    }
  }
}

void Network::initializeWeights() const {
  for (auto layer : layers) {
    if (layer->previousLayer != nullptr) {
      for (auto &n : layer->neurons) {
        n.initWeights(layer->previousLayer->neurons.size());
      }
    }
  }
}

void Network::initializeLayers() {
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

  bindLayers();
  initializeWeights();
  initializeNeighbors();
}

void Network::initializeNeighbors() {
  const auto &network_params = Manager::getInstance().network_params;
  for (auto &layer : layers) {
    int layer_size_x, layer_size_y;

    if (dynamic_cast<InputLayer *>(layer)) {
      layer_size_x = network_params.input_size_x;
      layer_size_y = network_params.input_size_y;
    } else if (dynamic_cast<HiddenLayer *>(layer)) {
      layer_size_x = network_params.hidden_size_x;
      layer_size_y = network_params.hidden_size_y;
    } else if (dynamic_cast<OutputLayer *>(layer)) {
      layer_size_x = network_params.output_size_x;
      layer_size_y = network_params.output_size_y;
    }

    for (size_t i = 0; i < layer->neurons.size(); ++i) {
      Neuron &neuron = layer->neurons[i];

      // Compute the coordinates of the neuron in the 2D grid
      int x = i % layer_size_x;
      int y = i / layer_size_x;

      // For each possible direction (up, down, left, right), check if there
      // is a neuron in that direction and, if so, establish a connection
      std::vector<std::pair<int, int>> directions = {
          {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
      for (auto [dx, dy] : directions) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < layer_size_x && ny >= 0 && ny < layer_size_y) {
          int ni = ny * layer_size_x + nx;
          Neuron &neighbor = layer->neurons[ni];

          // Create a connection with a random initial weight
          std::random_device rd;
          std::mt19937 gen(rd());
          std::uniform_real_distribution<float> dist(0.0f, 1.0f);
          RGBA weight = {dist(gen), dist(gen), dist(gen), dist(gen)};

          neuron.neighbors.push_back(Connection(&neighbor, weight));
        }
      }
    }
  }
}

void Network::SetActivationFunction(Layer *layer,
                                    EActivationFunction activation_function,
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
