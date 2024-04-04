#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportFacade.h"
#include "exception/ImportExportException.h"
#include <Manager.h>
#include <exception>
#include <memory>

using namespace sipai;

std::unique_ptr<NeuralNetwork> NeuralNetworkImportExportFacade::importModel() {
  try {
    auto network = jsonIE.importModel();
    return network;
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}

void NeuralNetworkImportExportFacade::importWeights(
    std::unique_ptr<NeuralNetwork> &network) {
  try {
    csvIE.importNeuronsWeights(network);
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}

void NeuralNetworkImportExportFacade::exportModel() const {
  try {
    jsonIE.exportModel();
    csvIE.exportNeuronsWeights();
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}