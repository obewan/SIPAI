/**
 * @file TrainingDataReader.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief TrainingDataFileReaderCSV
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"
#include "Data.h"

namespace sipai {
class TrainingDataReader {
public:
  /**
   * @brief Reads the training data from a CSV file.
   * @return A vector of data.
   */
  std::vector<Data> loadTrainingDataPaths();

  /**
   * @brief Reads the training data from a target folder.
   * @return A vector of data.
   */
  std::vector<Data> loadTrainingDataFolder();
};
} // namespace sipai