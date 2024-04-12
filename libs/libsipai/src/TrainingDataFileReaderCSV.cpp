#include "TrainingDataFileReaderCSV.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "csv_parser.h"
#include "exception/FileReaderException.h"
#include <fstream>
#include <memory>
#include <string>
// for csv_parser doc, see https://github.com/ashaduri/csv-parser

using namespace sipai;

std::unique_ptr<std::vector<ImagePathPair>>
TrainingDataFileReaderCSV::loadTrainingDataPaths() {
  const auto &trainingDataFile =
      Manager::getInstance().app_params.training_data_file;
  if (trainingDataFile.empty()) {
    throw FileReaderException("empty file path");
  }

  // Create a file stream object using RAII (Resource Acquisition Is
  // Initialization)
  std::ifstream file(trainingDataFile);
  if (!file.is_open()) {
    throw FileReaderException("Failed to open file: " + trainingDataFile);
  }

  auto trainingData = std::make_unique<std::vector<ImagePathPair>>();
  Csv::Parser csvParser;
  std::vector<std::vector<Csv::CellReference>> cell_refs;
  std::string line;
  int lineNumber = 1;
  while (std::getline(file, line)) {
    try {
      if (line.empty()) {
        continue;
      }
      std::string_view data(line);
      csvParser.parseTo2DVector(data, cell_refs);
      if (cell_refs.empty() || cell_refs.size() != 2) {
        throw FileReaderException("invalid column numbers, at line " +
                                  std::to_string(lineNumber));
      }
      std::pair<std::string, std::string> columns;
      columns.first = cell_refs[0][0].getCleanString().value();
      columns.second = cell_refs[1][0].getCleanString().value();
      trainingData->push_back(columns);
      lineNumber++;
    } catch (Csv::ParseError &ex) {
      SimpleLogger::LOG_ERROR("CSV parsing error at line (", lineNumber,
                              "): ", ex.what());
    }
  }

  // The file stream object will be automatically closed when it goes out of
  // scope

  return trainingData;
}