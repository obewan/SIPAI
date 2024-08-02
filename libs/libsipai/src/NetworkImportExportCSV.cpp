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

void NeuralNetworkImportExportCSV::exportNeuronsWeights(
    const std::unique_ptr<NeuralNetwork> &network, const AppParams &appParams,
    std::function<void(int)> progressCallback, int progressInitialValue) const {
  // get the csv filename
  std::string filename = Common::getFilenameCsv(appParams.network_to_export);
  std::ofstream file(filename);

  // Write the data
  size_t max_weights = network->max_weights;
  for (size_t layer_index = 0; layer_index < network->layers.size();
       layer_index++) {
    const auto &layer = network->layers.at(layer_index);
    if (layer->layerType == LayerType::LayerInput) {
      // no weights for Input Layer, as it will be input data weights
      continue;
    }

    for (size_t row = 0; row < layer->neurons.size(); row++) {
      for (size_t col = 0; col < layer->neurons.at(row).size(); col++) {
        Neuron &neuron = layer->neurons[row][col];
        // Write the neuron weights, empty 3rd neighbors column then
        file << layer_index << "," << neuron.weights.rows << ","
             << neuron.weights.cols << "," << row << "," << col << ",,"
             << neuron.toStringCsv(max_weights) << "\n";
        // Write the neighbors connections weights
        file << layer_index << "," << neuron.weights.rows << ","
             << neuron.weights.cols << "," << row << "," << col << ","
             << neuron.neighbors.size() << ","
             << neuron.toNeighborsStringCsv(max_weights) << "\n";
      }
    }
  }
}

void NeuralNetworkImportExportCSV::importNeuronsWeights(
    std::unique_ptr<NeuralNetwork> &network, const AppParams &appParams,
    std::function<void(int)> progressCallback, int progressInitialValue) const {

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
  std::string filename = Common::getFilenameCsv(appParams.network_to_import);

  // get the total of lines to use for progress calculus
  size_t totalLines = 0;
  if (progressCallback) {
    totalLines = Common::countLines(filename);
  }

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw ImportExportException("Failed to open file: " + filename);
  }

  // parsing the csv
  std::string line;
  int oldProgressValue = progressInitialValue;
  for (int current_line_number = 1; std::getline(file, line);
       ++current_line_number) {
    const auto &fields = split(line, ',');

    if (fields.size() < 6) {
      throw ImportExportException("CSV parsing error at line (" +
                                  std::to_string(current_line_number) +
                                  "): invalid column numbers");
    }

    auto layer_index = static_cast<size_t>(fields[0].value());
    auto weights_rows = static_cast<size_t>(fields[1].value());
    auto weights_cols = static_cast<size_t>(fields[2].value());
    auto neuron_row = static_cast<size_t>(fields[3].value());
    auto neuron_col = static_cast<size_t>(fields[4].value());
    auto neighboors_count = fields[5];

    if (!neighboors_count) {
      // add the neuron weights
      cv::Mat weights = cv::Mat((int)weights_rows, (int)weights_cols, CV_32FC4);
      size_t i_cols = 0;
      size_t i_rows = 0;
      for (size_t pos = 6; pos + 4 < fields.size();
           pos += 4) { // pos start at fields[6], then increment of
                       // the length of cv::Vec4f (4)
        auto r = fields[pos];
        auto g = fields[pos + 1];
        auto b = fields[pos + 2];
        auto a = fields[pos + 3];
        if (r && g && b && a) {
          weights.at<cv::Vec4f>((int)i_rows, (int)i_cols) =
              cv::Vec4f(*r, *g, *b, *a);
        }
        i_cols++;
        if (i_cols >= weights_cols) {
          i_cols = 0;
          i_rows++;
        }
      }
      network->layers.at(layer_index)
          ->neurons.at(neuron_row)
          .at(neuron_col)
          .weights = weights;
    } else {
      // add the neighboors and their weights
      // add the neuron weights
      std::vector<cv::Vec4f> weights;
      for (size_t pos = 6; pos + 4 < fields.size();
           pos += 4) { // pos start at fields[6], then increment of
                       // the length of cv::Vec4f (4)
        auto r = fields[pos];
        auto g = fields[pos + 1];
        auto b = fields[pos + 2];
        auto a = fields[pos + 3];
        if (r && g && b && a) {
          weights.push_back(cv::Vec4f(*r, *g, *b, *a));
        }
      }
      auto &connections = network->layers.at(layer_index)
                              ->neurons[neuron_row][neuron_col]
                              .neighbors;
      if (connections.size() != weights.size()) {
        throw ImportExportException("CSV parsing error at line (" +
                                    std::to_string(current_line_number) +
                                    "): invalid column numbers");
      }
      for (size_t i = 0; i < connections.size(); i++) {
        connections.at(i).weight = weights.at(i);
      }
    }

    if (progressCallback) {
      int value = progressInitialValue +
                  ((100 * current_line_number) / (int)totalLines);
      if (value != oldProgressValue) {
        progressCallback(value);
        oldProgressValue = value;
      }
    }
  }
}
