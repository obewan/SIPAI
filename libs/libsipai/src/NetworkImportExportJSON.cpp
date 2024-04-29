#include "AppParams.h"
#include "LayerHidden.h"
#include "LayerInput.h"
#include "LayerOutput.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportJSON.h"
#include "NeuralNetworkParams.h"
#include "SimpleLogger.h"
#include "exception/ImportExportException.h"
#include "json.hpp"
#include <fstream>
#include <memory>
// for nlohmann json doc, see https://github.com/nlohmann/json

using namespace sipai;

void NeuralNetworkImportExportJSON::exportModel(
    const std::unique_ptr<NeuralNetwork> &network,
    const NeuralNetworkParams &networkParams,
    const AppParams &appParams) const {
  using json = nlohmann::json;
  json json_network;

  // Serialize the version
  json_network["version"] = appParams.version;

  // Serialize the layers to JSON.
  for (auto layer : network->layers) {
    json json_layer = {{"type", layer->getLayerTypeStr()},
                       {"size_x", layer->size_x},
                       {"size_y", layer->size_y},
                       {"neurons", layer->neurons.size()}};
    json_network["layers"].push_back(json_layer);
  }

  // max weights info
  json_network["max_weights"] = network->max_weights;

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
  json_network["parameters"]["adaptive_learning_rate"] =
      json(networkParams.adaptive_learning_rate);
  json_network["parameters"]["adaptive_learning_rate_factor"] =
      json(networkParams.adaptive_learning_rate_factor);
  json_network["parameters"]["enable_adaptive_increase"] =
      json(networkParams.enable_adaptive_increase);
  json_network["parameters"]["error_min"] = json(networkParams.error_min);
  json_network["parameters"]["error_max"] = json(networkParams.error_max);
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

std::unique_ptr<NeuralNetwork>
NeuralNetworkImportExportJSON::importModel(const AppParams &appParams,
                                           NeuralNetworkParams &networkParams) {
  using json = nlohmann::json;
  const auto &logger = SimpleLogger::getInstance();

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
    auto network = std::make_unique<NeuralNetwork>();

    if (std::string jversion = json_model["version"];
        jversion != appParams.version) {
      logger.warn("The model version of the file is different from the current "
                  "version: " +
                  jversion + " vs " + appParams.version);
    }

    // Create a new Network object and deserialize the JSON data into it.
    networkParams.input_size_x = json_model["parameters"]["input_size_x"];
    networkParams.input_size_y = json_model["parameters"]["input_size_y"];
    networkParams.hidden_size_x = json_model["parameters"]["hidden_size_x"];
    networkParams.hidden_size_y = json_model["parameters"]["hidden_size_y"];
    networkParams.output_size_x = json_model["parameters"]["output_size_x"];
    networkParams.output_size_y = json_model["parameters"]["output_size_y"];
    networkParams.hiddens_count = json_model["parameters"]["hiddens_count"];
    networkParams.learning_rate = json_model["parameters"]["learning_rate"];
    networkParams.adaptive_learning_rate =
        json_model["parameters"]["adaptive_learning_rate"];
    networkParams.adaptive_learning_rate_factor =
        json_model["parameters"]["adaptive_learning_rate_factor"];
    networkParams.enable_adaptive_increase =
        json_model["parameters"]["enable_adaptive_increase"];
    networkParams.error_min = json_model["parameters"]["error_min"];
    networkParams.error_max = json_model["parameters"]["error_max"];
    networkParams.hidden_activation_alpha =
        json_model["parameters"]["hidden_activation_alpha"];
    networkParams.output_activation_alpha =
        json_model["parameters"]["output_activation_alpha"];
    networkParams.hidden_activation_function =
        json_model["parameters"]["hidden_activation_function"];
    networkParams.output_activation_function =
        json_model["parameters"]["output_activation_function"];

    network->max_weights = json_model["max_weights"];

    for (auto json_layer : json_model["layers"]) {
      // Get the type of the layer.
      std::string layer_type_str = json_layer["type"];
      LayerType layer_type = layer_map.at(layer_type_str);

      // // Create a new layer object of the appropriate type.
      Layer *layer = nullptr;
      switch (layer_type) {
      case LayerType::LayerInput:
        layer = new LayerInput((size_t)json_layer["size_x"],
                               (size_t)json_layer["size_y"]);
        break;
      case LayerType::LayerHidden:
        layer = new LayerHidden((size_t)json_layer["size_x"],
                                (size_t)json_layer["size_y"]);
        layer->eactivationFunction = networkParams.hidden_activation_function;
        layer->activationFunctionAlpha = networkParams.hidden_activation_alpha;
        break;
      case LayerType::LayerOutput:
        layer = new LayerOutput((size_t)json_layer["size_x"],
                                (size_t)json_layer["size_y"]);
        layer->eactivationFunction = networkParams.output_activation_function;
        layer->activationFunctionAlpha = networkParams.output_activation_alpha;
        break;
      default:
        throw ImportExportException("Layer type not recognized");
      }

      // Add the layer to the network.
      network->layers.push_back(layer);
    }

    if (network->layers.front()->layerType != LayerType::LayerInput) {
      throw ImportExportException("Invalid input layer");
    }

    if (network->layers.back()->layerType != LayerType::LayerOutput) {
      throw ImportExportException("Invalid output layer");
    }
    return network;
  } catch (const nlohmann::json::parse_error &e) {
    throw ImportExportException("Json parsing error: " + std::string(e.what()));
  }
}
