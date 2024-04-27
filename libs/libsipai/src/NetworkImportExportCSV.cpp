#include "Common.h"
#include "Layer.h"
#include "NeuralNetworkImportExportCSV.h"
#include "NeuronConnection.h"
#include "exception/EmptyCellException.h"
#include "exception/ImportExportException.h"
#include <algorithm> // for std::transform
#include <cctype>    // for std::tolower
#include <cstddef>
#include <exception>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <string>

using namespace sipai;

void NeuralNetworkImportExportCSV::importNeuronsWeights(
    std::unique_ptr<NeuralNetwork> &network, const AppParams &appParams) const {

  auto split = [](const std::string &s, char delimiter) {
    std::vector<std::optional<float>> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(token.empty() ? std::nullopt
                                     : std::make_optional(std::stof(token)));
    }
    return tokens;
  };

  // get the csv filename
  std::string filename = getFilenameCsv(appParams.network_to_import);
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw ImportExportException("Failed to open file: " + filename);
  }

  // parsing the csv
  std::string line;
  for (int current_line_number = 1; std::getline(file, line);
       ++current_line_number) {
    const auto &fields = split(line, ',');

    if (fields.size() < 4) {
      throw ImportExportException("CSV parsing error at line (" +
                                  std::to_string(current_line_number) +
                                  "): invalid column numbers");
    }

    auto layer_index = static_cast<size_t>(fields[0].value_or(0));
    auto neuron_index = static_cast<size_t>(fields[1].value_or(0));
    std::optional<size_t> neighboors_count = fields[2];

    // TODO: Update for refactoring
    //  cv::Mat weights;
    //  for (size_t pos = 3; pos + 3 < fields.size(); pos += 4) {
    //    auto r = fields[pos];
    //    auto g = fields[pos + 1];
    //    auto b = fields[pos + 2];
    //    auto a = fields[pos + 3];
    //    if (r && g && b && a) {
    //      weights.emplace_back(*r, *g, *b, *a);
    //    }
    //  }

    // if (!neighboors_count) {
    //   // add the neuron weights
    //   network->layers.at(layer_index)
    //       ->neurons.at(neuron_index)
    //       .weights.swap(weights);
    // } else if (!weights.empty()) {
    //   // add the neighboors and their weights
    //   auto &connections =
    //       network->layers.at(layer_index)->neurons.at(neuron_index).neighbors;
    //   if (connections.size() != weights.size()) {
    //     throw ImportExportException("CSV parsing error at line (" +
    //                                 std::to_string(current_line_number) +
    //                                 "): invalid column numbers");
    //   }
    //   std::transform(connections.begin(), connections.end(), weights.begin(),
    //                  connections.begin(),
    //                  [](NeuronConnection &connection, const RGBA &weight) {
    //                    connection.weight = weight;
    //                    return connection;
    //                  });
    // }
  }
}

void NeuralNetworkImportExportCSV::exportNeuronsWeights(
    const std::unique_ptr<NeuralNetwork> &network,
    const AppParams &appParams) const {
  // get the csv filename
  std::string filename = getFilenameCsv(appParams.network_to_export);
  std::ofstream file(filename);

  // Write the data
  // TODO: update for refactoring
  // size_t max_weights = network->max_weights;
  // for (size_t layer_index = 0; layer_index < network->layers.size();
  //      layer_index++) {
  //   const auto &layer = network->layers.at(layer_index);
  //   if (layer->layerType == LayerType::LayerInput) {
  //     // no weights for Input Layer, as it will be input data weights
  //     continue;
  //   }
  //   for (size_t neuron_index = 0; neuron_index < layer->neurons.size();
  //        neuron_index++) {
  //     const auto &neuron = layer->neurons.at(neuron_index);
  //     // Write the neuron weights, empty 3rd neighbors column then
  //     file << layer_index << "," << neuron_index << ",,"
  //          << neuron.toStringCsv(max_weights) << "\n";
  //     // Write the neighbors connections weights
  //     file << layer_index << "," << neuron_index << ","
  //          << neuron.neighbors.size() << ","
  //          << neuron.toNeighborsStringCsv(max_weights) << "\n";
  //   }
  // }
}