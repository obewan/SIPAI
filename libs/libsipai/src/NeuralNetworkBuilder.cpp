#include "NeuralNetworkBuilder.h"
#include "Common.h"
#include "Layer.h"
#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Manager.h"
#include "NeuralNetworkImportExportFacade.h"
#include "SimpleLogger.h"
#include "exception/NeuralNetworkException.h"
#include <cstddef>
#include <filesystem>
#include <opencv2/core/matx.hpp>

using namespace sipai;

NeuralNetworkBuilder::NeuralNetworkBuilder()
    : app_params_(Manager::getInstance().app_params),
      network_params_(Manager::getInstance().network_params) {}

NeuralNetworkBuilder::NeuralNetworkBuilder(AppParams &appParams,
                                           NeuralNetworkParams &networkParams)
    : app_params_(appParams), network_params_(networkParams) {}

NeuralNetworkBuilder &NeuralNetworkBuilder::createOrImport() {
  if (!app_params_.network_to_import.empty() &&
      std::filesystem::exists(app_params_.network_to_import)) {
    NeuralNetworkImportExportFacade neuralNetworkImportExport;
    SimpleLogger::LOG_INFO("Importing the neural network from ",
                           app_params_.network_to_import, "...");
    network_ =
        neuralNetworkImportExport.importModel(app_params_, network_params_);
    isImported = true;
  } else {
    SimpleLogger::LOG_INFO("Creating the neural network...");
    network_ = std::make_unique<NeuralNetwork>();
    isImported = false;
  }
  return *this;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::addLayers() {
  if (isImported) {
    return *this;
  }

  SimpleLogger::LOG_INFO("Adding layers...");
  if (!network_) {
    throw NeuralNetworkException("neural network null");
  }
  if (!network_->layers.empty()) {
    throw NeuralNetworkException("layers not empty");
  }

  // Add Input Layer
  auto inputLayer = new LayerInput(network_params_.input_size_x,
                                   network_params_.input_size_y);
  network_->layers.push_back(inputLayer);

  // Add Hidden Layers
  for (size_t i = 0; i < network_params_.hiddens_count; ++i) {
    auto hiddenLayer = new LayerHidden(network_params_.hidden_size_x,
                                       network_params_.hidden_size_y);
    hiddenLayer->eactivationFunction =
        network_params_.hidden_activation_function;
    hiddenLayer->activationFunctionAlpha =
        network_params_.hidden_activation_alpha;
    network_->layers.push_back(hiddenLayer);
  }

  // Add Output Layer
  auto outputLayer = new LayerOutput(network_params_.output_size_x,
                                     network_params_.output_size_y);
  outputLayer->eactivationFunction = network_params_.output_activation_function;
  outputLayer->activationFunctionAlpha =
      network_params_.output_activation_alpha;
  network_->layers.push_back(outputLayer);
  return *this;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::addNeighbors() {
  SimpleLogger::LOG_INFO("Adding neurons neighbors connections...");
  if (!network_) {
    throw NeuralNetworkException("neural network null");
  }
  if (network_->layers.empty()) {
    throw NeuralNetworkException("empty layers");
  }

  // For each possible direction (up, down, left, right), check if there
  // is a neuron in that direction and, if so, establish a connection
  std::vector<std::pair<int, int>> directions = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}};

  for (auto layer : network_->layers) {
    if (layer->layerType == LayerType::LayerInput) {
      continue;
    }
    for (auto &rows : layer->neurons) {
      for (auto &neuron : rows) {
        size_t pos_x = neuron.index_x;
        size_t pos_y = neuron.index_y;
        for (auto [dx, dy] : directions) {
          int nx = static_cast<int>(pos_x) + dx;
          int ny = static_cast<int>(pos_y) + dy;

          if (nx >= 0 && nx < static_cast<int>(layer->size_x) && ny >= 0 &&
              ny < static_cast<int>(layer->size_y)) {
            Neuron &neighbor = layer->neurons[ny][nx];

            cv::Vec4f weight;
            if (isImported) {
              weight = cv::Vec4f::zeros();
            } else {
              cv::Matx41f temp = cv::Matx41f::randn(0.0, 1.0);
              weight =
                  cv::Vec4f(temp.val[0], temp.val[1], temp.val[2], temp.val[3]);
            }

            neuron.neighbors.push_back(NeuronConnection(&neighbor, weight));
          }
        }
      }
    }
  }

  return *this;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::bindLayers() {
  SimpleLogger::LOG_INFO("Binding layers...");
  if (!network_) {
    throw NeuralNetworkException("neural network null");
  }
  if (network_->layers.empty()) {
    throw NeuralNetworkException("empty layers");
  }
  for (size_t i = 0; i < network_->layers.size(); ++i) {
    if (i > 0) {
      network_->layers.at(i)->previousLayer = network_->layers.at(i - 1);
    }
    if (i < network_->layers.size() - 1) {
      network_->layers.at(i)->nextLayer = network_->layers.at(i + 1);
    }
  }
  return *this;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::initializeWeights() {
  if (isImported) {
    NeuralNetworkImportExportFacade neuralNetworkImportExport;
    std::string filenameCsv = getFilenameCsv(app_params_.network_to_import);
    SimpleLogger::LOG_INFO("Importing layers neurons weights from ",
                           filenameCsv, "...");
    neuralNetworkImportExport.importWeights(network_, app_params_);
    return *this;
  }

  SimpleLogger::LOG_INFO("Initializing layers neurons weights...");
  if (!network_) {
    throw NeuralNetworkException("neural network null");
  }
  if (network_->layers.empty()) {
    throw NeuralNetworkException("empty layers");
  }
  network_->max_weights = 0;
  for (auto layer : network_->layers) {
    if (layer->previousLayer != nullptr) {
      for (auto &rows : layer->neurons) {
        for (auto &n : rows) {
          n.initWeights(layer->previousLayer->size_x,
                        layer->previousLayer->size_y);
          size_t new_size = layer->previousLayer->total();
          if (new_size > network_->max_weights) {
            network_->max_weights = new_size;
          }
        }
      }
    }
  }
  return *this;
}

NeuralNetworkBuilder &NeuralNetworkBuilder::setActivationFunction() {
  SimpleLogger::LOG_INFO("Setting neurons activation functions...");
  if (!network_) {
    throw NeuralNetworkException("neural network null");
  }
  if (network_->layers.empty()) {
    throw NeuralNetworkException("empty layers");
  }
  EActivationFunction activation_function;
  float activation_alpha = 0.0f;
  for (auto layer : network_->layers) {
    // Get the parameters
    switch (layer->layerType) {
    case LayerType::LayerInput:
      continue; // no activation function for input layer
    case LayerType::LayerHidden:
      activation_function = network_params_.hidden_activation_function;
      activation_alpha = network_params_.hidden_activation_alpha;
      break;
    case LayerType::LayerOutput:
      activation_function = network_params_.output_activation_function;
      activation_alpha = network_params_.output_activation_alpha;
      break;
    default:
      continue;
    }
    // Set the activation functions
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
      throw NeuralNetworkException("Unimplemented Activation Function");
    }
  }

  return *this;
}

std::unique_ptr<NeuralNetwork> NeuralNetworkBuilder::build() {
  return std::move(network_);
}
