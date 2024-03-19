#include "Manager.h"
#include "NeuralNetworkImportExportCSV.h"
#include "csv_parser.h"
#include "exception/EmptyCellException.h"
#include "exception/ImportExportException.h"
#include <algorithm> // for std::transform
#include <cctype>    // for std::tolower
#include <cstddef>
#include <exception>
#include <fstream>
#include <optional>
#include <regex> // for std::regex and std::regex_replace
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
  auto getRGBAValue = [](const std::vector<Csv::CellReference> &cells) {
    RGBA rgba;
    for (int i = 0; i < 4; ++i) {
      auto val = cells[i].getDouble();
      if (val.has_value()) {
        rgba.value[i] = static_cast<float>(val.value());
      } else {
        throw EmptyCellException(); // Or handle missing values differently
      }
    }
    return rgba;
  };

  // lambda function to add cell value to the weights vector
  auto processCell = [&getRGBAValue](
                         std::vector<RGBA> &weights,
                         const std::vector<Csv::CellReference> &cells) {
    try {
      weights.push_back({getRGBAValue(cells)}); // Use initializer list for RGBA
    } catch (EmptyCellException &) {
      // Ignore the exception caused by an empty cell (or handle differently)
      return;
    }
  };

  // get the csv filename
  const auto &appParams = Manager::getInstance().app_params;
  std::string filename = appParams.network_to_import;
  filename = std::regex_replace(
      filename, std::regex(".json$", std::regex::icase), ".csv");
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
      auto neighboor_index = getIndexValue(cell_refs.at(2));
      std::vector<RGBA> weights;

      if (layer_index && neuron_index) {
        std::for_each(cell_refs.begin() + 3, cell_refs.end(),
                      std::bind_front(processCell, std::ref(weights)));
        if (!neighboor_index) {
          // add the neuron weights
          network->layers.at(layer_index.value())
              ->neurons.at(neuron_index.value())
              .weights.swap(weights);
        } else if (weights.size() > 0) {
          // add the neighboor weight
          network->layers.at(layer_index.value())
              ->neurons.at(neuron_index.value())
              .neighbors.at(neighboor_index.value())
              .weight = weights.at(0);
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
  std::string filename = appParams.network_to_export;
  filename = std::regex_replace(
      filename, std::regex(".json$", std::regex::icase), ".csv");
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
  file << "Layer,Neuron,Neighbor";
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
          // RGBA column to fill the csv
          file << ",,,,";
        }
      }
      file << "\n";

      // Then write the neuron's neighbors weights
      file << layer_index << "," << neuron_index << ",";
      for (size_t i = 0; i < max_weights; ++i) {
        file << i << "";
        if (i < neuron.neighbors.size()) {
          file << "," << neuron.neighbors[i].weight.toStringCsv();
        } else {
          // If no neighbor write a blank RGBA column to fill the csv
          file << ",,,,";
        }
      }
      file << "\n";
    }
  }

  file.close();
}