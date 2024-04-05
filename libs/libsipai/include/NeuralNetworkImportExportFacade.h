/**
 * @file NeuralNetworkImportExportFacade.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Import/export of the neural network
 * @date 2024-02-20
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportCSV.h"
#include "NeuralNetworkImportExportJSON.h"
#include <memory>
namespace sipai {
class NeuralNetworkImportExportFacade {
public:
  virtual ~NeuralNetworkImportExportFacade() = default;

  /**
   * @brief Import a network model from JSON model file (without weights)
   *
   * @param appParams
   * @param networkParams
   * @return std::unique_ptr<NeuralNetwork>
   */
  virtual std::unique_ptr<NeuralNetwork>
  importModel(const AppParams &appParams, NeuralNetworkParams &networkParams);

  /**
   * @brief Import the network weights from a CSV weights file (network model
   * should be imported first)
   *
   * @param network
   * @param appParams
   */
  void importWeights(std::unique_ptr<NeuralNetwork> &network,
                     const AppParams &appParams);

  /**
   * @brief Export a network model files (JSON meta data and CSV neurons data)
   *
   * @param network
   * @param networkParams
   * @param appParams
   */
  virtual void exportModel(const std::unique_ptr<NeuralNetwork> &network,
                           const NeuralNetworkParams &networkParams,
                           const AppParams &appParams) const;

protected:
  NeuralNetworkImportExportCSV csvIE;
  NeuralNetworkImportExportJSON jsonIE;
};
} // namespace sipai