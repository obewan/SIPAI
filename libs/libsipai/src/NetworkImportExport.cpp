#include "NeuralNetwork.h"
#include "NeuralNetworkImportExport.h"
#include "NeuralNetworkImportExportCSV.h"
#include "NeuralNetworkImportExportJSON.h"
#include "exception/ImportExportException.h"
#include <Manager.h>
#include <exception>
#include <memory>

using namespace sipai;

std::unique_ptr<NeuralNetwork> NeuralNetworkImportExport::importModel() {
  try {
    NeuralNetworkImportExportCSV NIE_CSV;
    NeuralNetworkImportExportJSON NIE_JSON;
    auto network = NIE_JSON.importModel();
    NIE_CSV.importNeuronsWeights(network);
    return network;
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}

void NeuralNetworkImportExport::exportModel() const {
  try {
    NeuralNetworkImportExportCSV NIE_CSV;
    NeuralNetworkImportExportJSON NIE_JSON;
    NIE_JSON.exportModel();
    NIE_CSV.exportNeuronsWeights();
  } catch (std::exception &ex) {
    throw ImportExportException(ex.what());
  }
}