#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportFacade.h"
#include "exception/ImportExportException.h"
#include <exception>
#include <memory>

using namespace sipai;

std::unique_ptr<NeuralNetwork> NeuralNetworkImportExportFacade::importModel(
    const AppParams &appParams, NeuralNetworkParams &networkParams) {
  try {
    auto network = jsonIE.importModel(appParams, networkParams);
    return network;
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}

void NeuralNetworkImportExportFacade::importWeights(
    std::unique_ptr<NeuralNetwork> &network, const AppParams &appParams,
    std::function<void(int)> progressCallback, int progressInitialValue) {
  try {
    csvIE.importNeuronsWeights(network, appParams, progressCallback,
                               progressInitialValue);
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}

void NeuralNetworkImportExportFacade::exportModel(
    const std::unique_ptr<NeuralNetwork> &network,
    const NeuralNetworkParams &networkParams,
    const AppParams &appParams) const {
  try {
    jsonIE.exportModel(network, networkParams, appParams);
    csvIE.exportNeuronsWeights(network, appParams);
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}