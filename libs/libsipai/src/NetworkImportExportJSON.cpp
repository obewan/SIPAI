#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "Manager.h"
#include "NeuralNetworkImportExportJSON.h"
#include "SimpleLogger.h"
#include "exception/ImportExportException.h"
#include "json.hpp"
#include <fstream>
// for nlohmann json doc, see https://github.com/nlohmann/json

using namespace sipai;

std::unique_ptr<NeuralNetwork> NeuralNetworkImportExportJSON::importModel() {
  using json = nlohmann::json;
  const auto &logger = SimpleLogger::getInstance();
  const auto &appParams = Manager::getInstance().app_params;

  if (appParams.network_to_import.empty()) {
    throw ImportExportException("Empty parameter network_to_import");
  }

  std::string path_in_ext = appParams.network_to_import;
  if (std::filesystem::path p(path_in_ext); p.parent_path().empty()) {
    path_in_ext = "./" + path_in_ext;
  }

  std::ifstream file(path_in_ext);
  json json_model;

  if (!file.is_open()) {
    throw ImportExportException("Failed to open file: " + path_in_ext);
  }

  if (!json::accept(file)) {
    file.close();
    throw ImportExportException("Json parsing error: " + path_in_ext);
  }
  file.seekg(0, std::ifstream::beg);

  try {
    json_model = json::parse(file);

    if (std::string jversion = json_model["version"];
        jversion != appParams.version) {
      logger.warn("The model version of the file is different from the current "
                  "version: " +
                  jversion + " vs " + appParams.version);
    }

    // Create a new Network object and deserialize the JSON data into it.
    auto model = std::make_unique<NeuralNetwork>();
    auto params = Manager::getInstance().network_params;
    params.input_size_x = json_model["parameters"]["input_size_x"];
    params.input_size_y = json_model["parameters"]["input_size_y"];
    params.hidden_size_x = json_model["parameters"]["hidden_size_x"];
    params.hidden_size_y = json_model["parameters"]["hidden_size_y"];
    params.output_size_x = json_model["parameters"]["output_size_x"];
    params.output_size_y = json_model["parameters"]["output_size_y"];
    params.hiddens_count = json_model["parameters"]["hiddens_count"];
    params.learning_rate = json_model["parameters"]["learning_rate"];
    params.hidden_activation_alpha =
        json_model["parameters"]["hidden_activation_alpha"];
    params.output_activation_alpha =
        json_model["parameters"]["output_activation_alpha"];
    params.hidden_activation_function =
        json_model["parameters"]["hidden_activation_function"];
    params.output_activation_function =
        json_model["parameters"]["output_activation_function"];

    for (auto json_layer : json_model["layers"]) {
      // Get the type of the layer.
      std::string layer_type_str = json_layer["type"];
      LayerType layer_type = layer_map.at(layer_type_str);
      int layer_size_x, layer_size_y;

      // Create a new layer object of the appropriate type.
      Layer *layer = nullptr;
      switch (layer_type) {
      case LayerType::LayerInput:
        layer = new LayerInput();
        layer_size_x = params.input_size_x;
        layer_size_y = params.input_size_y;
        break;
      case LayerType::LayerHidden:
        layer = new LayerHidden();
        layer_size_x = params.hidden_size_x;
        layer_size_y = params.hidden_size_y;
        break;
      case LayerType::LayerOutput:
        layer = new LayerOutput();
        layer_size_x = params.output_size_x;
        layer_size_y = params.output_size_y;
        break;
      default:
        throw ImportExportException("Layer type not recognized");
      }

      // Add neurons and their neighbors without their weights
      layer->neurons = std::vector<Neuron>((size_t)json_layer["neurons"]);
      for (size_t i = 0; i < layer->neurons.size(); ++i) {
        model->addNeuronNeighbors(layer->neurons[i], layer, i, layer_size_x,
                                  layer_size_y, false);
      }

      // Set activation functions
      switch (layer->layerType) {
      case LayerType::LayerInput: // no activation function here
        break;
      case LayerType::LayerHidden:
        model->SetActivationFunction(layer, params.hidden_activation_function,
                                     params.hidden_activation_alpha);
        break;
      case LayerType::LayerOutput:
        model->SetActivationFunction(layer, params.output_activation_function,
                                     params.output_activation_alpha);
        break;
      default:
        throw ImportExportException("Layer type not recognized");
      }

      // Add the layer to the network.
      model->layers.push_back(layer);
    }

    if (model->layers.front()->layerType != LayerType::LayerInput) {
      throw ImportExportException("Invalid input layer");
    }

    if (model->layers.back()->layerType != LayerType::LayerOutput) {
      throw ImportExportException("Invalid output layer");
    }

    model->bindLayers();
    return model;

  } catch (const nlohmann::json::parse_error &e) {
    throw ImportExportException("Json parsing error: " + std::string(e.what()));
  }
}

void NeuralNetworkImportExportJSON::exportModel() const {
  using json = nlohmann::json;
  json json_network;
  auto &network = Manager::getInstance().network;
  auto &networkParams = Manager::getInstance().network_params;
  const auto &appParams = Manager::getInstance().app_params;

  // Serialize the version
  json_network["version"] = appParams.version;

  // Serialize the layers to JSON.
  for (auto layer : network->layers) {
    json json_layer = {{"type", layer->getLayerTypeStr()},
                       {"neurons", layer->neurons.size()}};
    json_network["layers"].push_back(json_layer);
  }

  // Serialize the parameters to JSON.
  json_network["parameters"]["input_size_x"] = json(networkParams.input_size_x);
  json_network["parameters"]["input_size_y"] = json(networkParams.input_size_y);
  json_network["parameters"]["hidden_size_x"] =
      json(networkParams.hidden_size_x);
  json_network["parameters"]["hidden_size_y"] =
      json(networkParams.hidden_size_y);
  json_network["parameters"]["output_size_x"] =
      json(networkParams.output_size_x);
  json_network["parameters"]["output_size_y"] =
      json(networkParams.output_size_y);
  json_network["parameters"]["hiddens_count"] =
      json(networkParams.hiddens_count);
  json_network["parameters"]["learning_rate"] =
      json(networkParams.learning_rate);
  json_network["parameters"]["hidden_activation_alpha"] =
      json(networkParams.hidden_activation_alpha);
  json_network["parameters"]["output_activation_alpha"] =
      json(networkParams.output_activation_alpha);
  json_network["parameters"]["hidden_activation_function"] =
      json(networkParams.hidden_activation_function);
  json_network["parameters"]["output_activation_function"] =
      json(networkParams.output_activation_function);

  // Write the JSON object to the file.
  // The 4 argument specifies the indentation level of the resulting string.
  std::ofstream file(appParams.network_to_export);
  file << json_network.dump(2);
  file.close();
}