#include "TrainingDataFileReaderCSV.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "csv_parser.h"
#include "exception/FileReaderException.h"
#include <fstream>
// for csv_parser doc, see https://github.com/ashaduri/csv-parser

using namespace sipai;

TrainingData TrainingDataFileReaderCSV::getTrainingData() {
  const auto &trainingDataFile =
      Manager::getInstance().app_params.trainingDataFile;
  if (trainingDataFile.empty()) {
    throw FileReaderException("empty file path");
  }

  // Create a file stream object using RAII (Resource Acquisition Is
  // Initialization)
  std::ifstream file(trainingDataFile);
  if (!file.is_open()) {
    throw FileReaderException("Failed to open file: " + trainingDataFile);
  }

  TrainingData trainingData;
  Csv::Parser csvParser;
  std::vector<std::vector<Csv::CellReference>> cell_refs;
  std::string line;
  int lineNumber = 1;
  while (std::getline(file, line)) {
    try {
      std::string_view data(line);
      csvParser.parseTo2DVector(data, cell_refs);
      if (cell_refs.empty() || cell_refs.size() != 2) {
        throw FileReaderException("invalide column numbers");
      }
      std::pair<std::string, std::string> columns;
      columns.first = cell_refs[0][0].getCleanString().value();
      columns.second = cell_refs[1][0].getCleanString().value();
      trainingData.push_back(columns);
      lineNumber++;
    } catch (Csv::ParseError &ex) {
      SimpleLogger::LOG_ERROR("CSV parse error at line (", lineNumber,
                              "): ", ex.what());
    }
  }

  // The file stream object will be automatically closed when it goes out of
  // scope

  return trainingData;
}