#include "Common.h"
#include "Layer.h"
#include "NeuralNetworkImportExportCSV.h"
#include "NeuronConnection.h"
#include "csv_parser.h"
#include "exception/EmptyCellException.h"
#include "exception/ImportExportException.h"
#include <algorithm> // for std::transform
#include <cctype>    // for std::tolower
#include <cstddef>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
// for csv_paser doc, see https://github.com/ashaduri/csv-parser

using namespace sipai;

void NeuralNetworkImportExportCSV::importNeuronsWeights(
    std::unique_ptr<NeuralNetwork> &network, const AppParams &appParams) const {

  // get the csv filename
  std::string filename = getFilenameCsv(appParams.network_to_import);
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw ImportExportException("Failed to open file: " + filename);
  }

  // parsing the csv
  Csv::Parser csv_parser;
  std::string line;
  int current_line_number = 0;
  while (std::getline(file, line)) {
    current_line_number++;
    std::vector<std::vector<Csv::CellReference>> cell_refs;

    try {
      std::string_view data(line);
      csv_parser.parseTo2DVector(data, cell_refs);
    } catch (Csv::ParseError &ex) {
      throw ImportExportException("CSV parsing error at line (" +
                                  std::to_string(current_line_number) +
                                  "): " + ex.what());
    }

    if (cell_refs.empty() || cell_refs.size() < 3) {
      throw ImportExportException("CSV parsing error at line (" +
                                  std::to_string(current_line_number) +
                                  "): invalid column numbers");
    }
    if (cell_refs.size() == 3) {
      continue;
    }

    try {
      auto layer_index = cell_refs.at(0).at(0).getDouble();
      auto neuron_index = cell_refs.at(1).at(0).getDouble();
      auto neighboors_count = cell_refs.at(2).at(0).getDouble();
      std::vector<RGBA> weights;

      if (layer_index && neuron_index) {
        for (size_t pos = 3; pos + 3 < cell_refs.size(); pos += 4) {
          auto r = cell_refs.at(pos).at(0).getDouble();
          auto g = cell_refs.at(pos + 1).at(0).getDouble();
          auto b = cell_refs.at(pos + 2).at(0).getDouble();
          auto a = cell_refs.at(pos + 3).at(0).getDouble();
          if (r.has_value() && g.has_value() && b.has_value() &&
              a.has_value()) {
            weights.push_back(RGBA(r.value(), g.value(), b.value(), a.value()));
          }
        }
        if (!neighboors_count) {
          // add the neuron weights
          network->layers.at(layer_index.value())
              ->neurons.at(neuron_index.value())
              .weights.swap(weights);
        } else if (weights.size() > 0) {
          // add the neighboors and their weights
          auto &connections = network->layers.at(layer_index.value())
                                  ->neurons.at(neuron_index.value())
                                  .neighbors;
          if (connections.size() != weights.size()) {
            throw ImportExportException("CSV parsing error at line (" +
                                        std::to_string(current_line_number) +
                                        "): invalid column numbers");
          }
          std::transform(connections.begin(), connections.end(),
                         weights.begin(), connections.begin(),
                         [](NeuronConnection &connection, const RGBA &weight) {
                           connection.weight = weight;
                           return connection;
                         });
        }
      }
    } catch (std::exception &ex) {
      throw ImportExportException(ex.what());
    }
  }

  file.close();
}

void NeuralNetworkImportExportCSV::exportNeuronsWeights(
    const std::unique_ptr<NeuralNetwork> &network,
    const AppParams &appParams) const {
  // get the csv filename
  std::string filename = getFilenameCsv(appParams.network_to_export);
  std::ofstream file(filename);

  // Determine the maximum number of weights any neuron has
  size_t max_weights = 0;
  for (const auto &layer : network->layers) {
    for (const auto &neuron : layer->neurons) {
      if (neuron.weights.size() > max_weights) {
        max_weights = neuron.weights.size();
      }
    }
  }

  // Write the data
  for (size_t layer_index = 0; layer_index < network->layers.size();
       layer_index++) {
    const auto &layer = network->layers.at(layer_index);
    if (layer->layerType == LayerType::LayerInput) {
      // no weights for Input Layer, as it will be input data weights
      continue;
    }
    for (size_t neuron_index = 0; neuron_index < layer->neurons.size();
         neuron_index++) {
      const auto &neuron = layer->neurons.at(neuron_index);

      // Write the layer index and neuron index to the CSV file, and an empty
      // neighbor index
      file << layer_index << "," << neuron_index << ",,";
      // Write the weights to the CSV file
      for (size_t i = 0; i < max_weights; ++i) {
        if (i < neuron.weights.size()) {
          file << neuron.weights[i].toStringCsv();
        } else {
          // If the neuron doesn't have a weight for this index, write a blank
          // RGBA 4 columns to fill the csv
          file << ",,,";
        }
        if (i < max_weights - 1) {
          file << ","; // add a separator except for last columns
        }
      }
      file << "\n";

      // Then write the neuron's neighbors connections weights on a single line
      // (to reduce csv file size)
      file << layer_index << "," << neuron_index << ","
           << neuron.neighbors.size() << ",";
      for (size_t i = 0; i < max_weights; ++i) {
        if (i < neuron.neighbors.size()) {
          file << neuron.neighbors[i].weight.toStringCsv();
        } else {
          // If no more neighbor connection write a blank RGBA 4 columns to
          // fill the csv
          file << ",,,";
        }
        if (i < max_weights - 1) {
          file << ","; // add a separator except for last columns
        }
      }
      file << "\n";
    }
  }

  file.close();
}