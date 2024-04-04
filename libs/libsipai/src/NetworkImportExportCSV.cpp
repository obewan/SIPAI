#include "Common.h"
#include "Manager.h"
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

using namespace sipai;

void NeuralNetworkImportExportCSV::importNeuronsWeights(
    std::unique_ptr<NeuralNetwork> &network) const {
  // lambda function to convert to float
  auto getIndexValue = [](const std::vector<Csv::CellReference> &cells)
      -> std::optional<size_t> {
    auto val = cells[0].getDouble();
    if (val.has_value()) {
      return static_cast<size_t>(val.value());
    } else {
      return std::nullopt;
    }
  };

  // lambda function to convert to RGBA floats
  auto getRGBAValue =
      [](const std::vector<Csv::CellReference> &cells) -> std::optional<RGBA> {
    RGBA rgba;
    for (int i = 0; i < 4; ++i) {
      auto val = cells[i].getDouble();
      if (val.has_value()) {
        rgba.value[i] = static_cast<float>(val.value());
      } else {
        return std::nullopt;
      }
    }
    return rgba;
  };

  // lambda function to add cell value to the weights vector
  auto processCell =
      [&getRGBAValue](std::vector<RGBA> &weights,
                      const std::vector<Csv::CellReference> &cells) {
        const auto &rgba = getRGBAValue(cells);
        if (rgba.has_value()) {
          weights.push_back(rgba.value());
        }
      };

  // get the csv filename
  const auto &appParams = Manager::getInstance().app_params;
  std::string filename = getFilenameCsv(appParams.network_to_import);
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw ImportExportException("Failed to open file: " + filename);
  }

  // parsing the csv
  Csv::Parser csv_parser;
  std::string line;
  int current_line_number = 0;
  bool header_skipped = false;
  while (std::getline(file, line)) {
    current_line_number++;
    if (!header_skipped) {
      header_skipped = true;
      continue;
    }
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
      auto layer_index = getIndexValue(cell_refs.at(0));
      auto neuron_index = getIndexValue(cell_refs.at(1));
      auto neighboors_count = getIndexValue(cell_refs.at(2));
      std::vector<RGBA> weights;

      if (layer_index && neuron_index) {
        std::for_each(cell_refs.begin() + 3, cell_refs.end(),
                      std::bind_front(processCell, std::ref(weights)));
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

void NeuralNetworkImportExportCSV::exportNeuronsWeights() const {
  // get the csv filename
  const auto &appParams = Manager::getInstance().app_params;
  std::string filename = getFilenameCsv(appParams.network_to_export);
  std::ofstream file(filename);

  // Determine the maximum number of weights any neuron has
  auto &network = Manager::getInstance().network;
  size_t max_weights = 0;
  for (const auto &layer : network->layers) {
    for (const auto &neuron : layer->neurons) {
      if (neuron.weights.size() > max_weights) {
        max_weights = neuron.weights.size();
      }
    }
  }

  // Write the header to the CSV file
  file << "Layer,Neuron,Neighbors";
  for (size_t i = 0; i < max_weights; ++i) {
    file << ",wR" << (i + 1) << ",wG" << (i + 1) << ",wB" << (i + 1) << ",wA"
         << (i + 1);
  }
  file << "\n";

  // Write the data
  for (size_t layer_index = 0; layer_index < network->layers.size();
       layer_index++) {
    const auto &layer = network->layers.at(layer_index);
    for (size_t neuron_index = 0; neuron_index < layer->neurons.size();
         neuron_index++) {
      const auto &neuron = layer->neurons.at(neuron_index);

      // Write the layer index and neuron index to the CSV file, and an empty
      // neighbor index
      file << layer_index << "," << neuron_index << ",,";
      // Write the weights to the CSV file
      for (size_t i = 0; i < max_weights; ++i) {
        if (i < neuron.weights.size()) {
          file << "," << neuron.weights[i].toStringCsv();
        } else {
          // If the neuron doesn't have a weight for this index, write a blank
          // RGBA 4 columns to fill the csv
          file << ",,,,";
        }
      }
      file << "\n";

      // Then write the neuron's neighbors connections weights on a single line
      // (to reduce csv file size)
      file << layer_index << "," << neuron_index << ","
           << neuron.neighbors.size();
      for (size_t i = 0; i < max_weights; ++i) {
        if (i < neuron.neighbors.size()) {
          file << "," << neuron.neighbors[i].weight.toStringCsv();
        } else {
          // If no more neighbor connection write a blank RGBA 4 columns to
          // fill the csv
          file << ",,,,";
        }
      }
      file << "\n";
    }
  }

  file.close();
}