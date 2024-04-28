/**
 * @file NeuralNetworkImportExportJSON.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Json import/export of the network meta data
 * @see https://github.com/nlohmann/json
 * @date 2023-10-29
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "AppParams.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkParams.h"
#include <memory>

namespace sipai {
/**
 * @brief NetworkImportExportJSON class to export and import network models
 * using the JSON format.
 *
 */
class NeuralNetworkImportExportJSON {
public:
  /**
   * @brief Export a Network model into a json file.
   *
   * @param network
   * @param networkParams
   * @param appParams
   */
  void exportModel(const std::unique_ptr<NeuralNetwork> &network,
                   const NeuralNetworkParams &networkParams,
                   const AppParams &appParams) const;

  /**
   * @brief Parse a json file into a network model.
   *
   * @param appParams
   * @param networkParams
   * @return std::unique_ptr<NeuralNetwork>
   */
  std::unique_ptr<NeuralNetwork>
  importModel(const AppParams &appParams, NeuralNetworkParams &networkParams);
};
} // namespace sipai