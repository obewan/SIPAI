#include "NeuralNetwork.h"
#include "NeuralNetworkImportExport.h"
#include "NeuralNetworkImportExportCSV.h"
#include "NeuralNetworkImportExportJSON.h"
#include "exception/ImportExportException.h"
#include <Manager.h>
#include <exception>
#include <memory>

using namespace sipai;

void NeuralNetworkImportExport::importModel() {
  try {
    NeuralNetworkImportExportCSV NIE_CSV;
    NeuralNetworkImportExportJSON NIE_JSON;
    Manager::getInstance().network = NIE_JSON.importModel();
    NIE_CSV.importNeuronsWeights();
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