/**
 * @file NetworkImportExportCSV.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief CSV import/export of the network neurons
 * @date 2024-02-20
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "AppParams.h"
#include "NeuralNetwork.h"

namespace sipai {
class NeuralNetworkImportExportCSV {
public:
  /**
   * @brief Import the network neurons data from a CSV file.
   */
  void importNeuronsWeights() const;

  /**
   * @brief Export the network neurons data to a CSV file.
   */
  void exportNeuronsWeights() const;
};
} // namespace sipai