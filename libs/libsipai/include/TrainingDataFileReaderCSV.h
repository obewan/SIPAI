/**
 * @file TrainingDataFileReaderCSV.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief TrainingDataFileReaderCSV
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"

namespace sipai {
class TrainingDataFileReaderCSV {
public:
  /**
   * @brief Reads the training data from a CSV file and returns a vector of
   * input-target file path pairs.
   *
   * This method reads the training data from a CSV file specified in the
   * application parameters. The CSV file should have two columns, where the
   * first column contains the input file paths, and the second column contains
   * the corresponding target file paths.
   *
   * The method performs the following steps:
   * 1. Retrieves the training data file path from the application parameters.
   * 2. Opens the file and reads its contents line by line.
   * 3. For each line, parses the input and target file paths using a CSV
   * parser.
   * 4. Stores the input-target file path pairs in a vector.
   * 5. Handles and logs any parsing errors that may occur.
   *
   * @throw FileReaderException If the file path is empty or the file cannot be
   * opened.
   * @throw FileReaderException If the CSV line has an invalid number of
   * columns.
   *
   * @return A vector of pairs, where each pair contains the input file path and
   * the corresponding target file path.
   */
  std::vector<std::unique_ptr<ImagePathPair>> loadTrainingDataPaths();
};
} // namespace sipai