/**
 * @file NeuralNetworkBuilder.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief NeuralNetwork Builder
 * @date 2024-03-30
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "AppParams.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportFacade.h"
#include <memory.h>
#include <memory>

namespace sipai {
class NeuralNetworkBuilder {
public:
  std::unique_ptr<NeuralNetwork> build();

private:
  std::unique_ptr<NeuralNetwork> network_ = nullptr;
};
} // namespace sipai
