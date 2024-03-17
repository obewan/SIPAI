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

namespace sipai {
class NeuralNetworkImportExport {
public:
  virtual ~NeuralNetworkImportExport() = default;

  /**
   * @brief Import a network model files (JSON meta data and CSV neurons data)
   */
  virtual void importModel();

  /**
   * @brief Export a network model files (JSON meta data and CSV neurons data)
   */
  virtual void exportModel() const;

protected:
};
} // namespace sipai