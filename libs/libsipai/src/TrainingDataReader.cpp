#include "TrainingDataReader.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "csv_parser.h"
#include "exception/FileReaderException.h"
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

// for csv_parser doc, see https://github.com/ashaduri/csv-parser

constexpr int MAX_ERRORS = 5;

using namespace sipai;

std::vector<Data> TrainingDataReader::loadTrainingDataPaths() {
  const auto &training_data_file =
      Manager::getInstance().app_params.training_data_file;
  if (training_data_file.empty()) {
    throw FileReaderException("empty file path");
  }

  // Create a file stream object using RAII (Resource Acquisition Is
  // Initialization)
  std::ifstream file(training_data_file);
  if (!file.is_open()) {
    throw FileReaderException("Failed to open file: " + training_data_file);
  }

  std::vector<Data> datas;
  Csv::Parser csvParser;
  std::vector<std::vector<Csv::CellReference>> cell_refs;
  std::string line;
  int lineNumber = 1;
  int totalErrors = 0;
  while (std::getline(file, line)) {
    try {
      if (line.empty()) {
        continue;
      }
      std::string_view str_view(line);
      csvParser.parseTo2DVector(str_view, cell_refs);
      if (cell_refs.empty() || cell_refs.size() != 2) {
        throw FileReaderException("invalid column numbers, at line " +
                                  std::to_string(lineNumber));
      }
      Data data;
      data.file_input = cell_refs[0][0].getCleanString().value();
      data.file_target = cell_refs[1][0].getCleanString().value();
      datas.push_back(data);
      lineNumber++;
    } catch (Csv::ParseError &ex) {
      totalErrors++;
      if (totalErrors < MAX_ERRORS) {
        SimpleLogger::LOG_ERROR("CSV parsing error at line (", lineNumber,
                                "): ", ex.what());
      } else {
        throw FileReaderException("Too many parsing errors.");
      }
    }
  }
  return datas;
}

std::vector<Data> TrainingDataReader::loadTrainingDataFolder() {
  const auto &training_data_folder =
      Manager::getInstance().app_params.training_data_folder;
  if (training_data_folder.empty()) {
    throw FileReaderException("empty folder path");
  }

  std::vector<Data> datas;
  // Add images paths from the folder
  for (const auto &entry :
       std::filesystem::directory_iterator(training_data_folder)) {
    if (entry.is_regular_file()) {
      // Convert the extension to lowercase
      std::string extension = entry.path().extension().string();
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     ::tolower);
      // Check if the file is an image by checking its extension
      if (valid_extensions.find(extension) != valid_extensions.end()) {
        Data data;
        data.file_target = entry.path().string();
        datas.push_back(data);
      }
    }
  }
  return datas;
}