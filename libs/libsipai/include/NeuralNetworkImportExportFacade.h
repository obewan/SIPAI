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
   */
  virtual std::unique_ptr<NeuralNetwork> importModel();

  /**
   * @brief Import the network weights from a CSV weights file (network model
   * should be imported first)
   *
   * @param network
   */
  void importWeights(std::unique_ptr<NeuralNetwork> &network);

  /**
   * @brief Export a network model files (JSON meta data and CSV neurons data)
   */
  virtual void exportModel() const;

protected:
  NeuralNetworkImportExportCSV csvIE;
  NeuralNetworkImportExportJSON jsonIE;
};
} // namespace sipai