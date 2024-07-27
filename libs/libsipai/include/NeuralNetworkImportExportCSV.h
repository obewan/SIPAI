/**
 * @file NeuralNetworkImportExportCSV.h
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
#include <memory>

namespace sipai {
class NeuralNetworkImportExportCSV {
public:
  /**
   * @brief Export the network neurons data to a CSV file.
   *
   * @param network
   * @param appParams
   * @param progressCallback
   */
  void exportNeuronsWeights(const std::unique_ptr<NeuralNetwork> &network,
                            const AppParams &appParams,
                            std::function<void(int)> progressCallback = {},
                            int progressInitialValue = 0) const;

  /**
   * @brief Import the network neurons data from a CSV file.
   * @param network
   * @param appParams
   * @param progressCallback
   */
  void importNeuronsWeights(std::unique_ptr<NeuralNetwork> &network,
                            const AppParams &appParams,
                            std::function<void(int)> progressCallback = {},
                            int progressInitialValue = 0) const;
};
} // namespace sipai