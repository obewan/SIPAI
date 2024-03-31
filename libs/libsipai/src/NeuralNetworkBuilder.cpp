#include "NeuralNetworkBuilder.h"
#include "Manager.h"
#include "NeuralNetworkImportExportFacade.h"
#include <filesystem>

using namespace sipai;

std::unique_ptr<NeuralNetwork> NeuralNetworkBuilder::build() {
  auto &manager = Manager::getInstance();
  const auto &appParams = manager.app_params;

  // import or create a new neural network
  if (!appParams.network_to_import.empty() &&
      std::filesystem::exists(appParams.network_to_import)) {
    NeuralNetworkImportExportFacade neuralNetworkImportExport;
    network_ = neuralNetworkImportExport.importModel();
  } else {
    network_ = std::make_unique<NeuralNetwork>();
    network_->initialize();
  }

  return std::move(network_);
}
