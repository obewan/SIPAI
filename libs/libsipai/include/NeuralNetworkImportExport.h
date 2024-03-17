/**
 * @file NetworkImportExport.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Import/export of the neural network
 * @date 2024-02-20
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "NeuralNetworkImportExportCSV.h"
#include "NeuralNetworkImportExportJSON.h"

namespace sipai {
class NeuralNetworkImportExport {
public:
  virtual ~NeuralNetworkImportExport() = default;

  /**
   * @brief Import a network model files (JSON meta data and CSV neurons data)
   *
   * @param app_params
   * @return Network*
   */
  virtual NeuralNetwork *importModel(const AppParams &app_params);

  /**
   * @brief Export a network model files (JSON meta data and CSV neurons data)
   *
   * @param network
   * @param app_params
   */
  virtual void exportModel(const NeuralNetwork *network,
                           const AppParams &app_params) const;

protected:
  NeuralNetworkImportExportCSV NIE_CSV;
  NeuralNetworkImportExportJSON NIE_JSON;
};
} // namespace sipai