#include "Manager.h"
#include "Common.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkBuilder.h"
#include "SimpleLogger.h"
#include "TrainingDataFileReaderCSV.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <vector>

using namespace sipai;

void Manager::createOrImportNetwork() {
  if (!network) {
    auto builder = std::make_unique<NeuralNetworkBuilder>();
    network = builder->createOrImport()
                  .addLayers()
                  .bindLayers()
                  .addNeighbors()
                  .initializeWeights()
                  .setActivationFunction()
                  .build();
  }
}

void Manager::exportNetwork() {
  if (!app_params.network_to_export.empty()) {
    SimpleLogger::LOG_INFO("Saving the neural network to ",
                           app_params.network_to_export, " and ",
                           getFilenameCsv(app_params.network_to_export), "...");
    auto exportator = std::make_unique<NeuralNetworkImportExportFacade>();
    exportator->exportModel(network, network_params, app_params);
  }
}

void Manager::run() {
  SimpleLogger::LOG_INFO(getVersionHeader());

  // Initialize network
  createOrImportNetwork();

  // Run with visitor
  switch (app_params.run_mode) {
  case ERunMode::TrainingMonitored:
    runWithVisitor(runnerVisitorFactory_.getTrainingMonitoredVisitor());
    break;
  case ERunMode::Enhancer:
    runWithVisitor(runnerVisitorFactory_.getEnhancerVisitor());
    break;
  default:
    break;
  }
}

void Manager::runWithVisitor(const RunnerVisitor &visitor) { visitor.visit(); }

std::unique_ptr<TrainingData> Manager::loadTrainingData() {
  return TrainingDataFileReaderCSV{}.getTrainingData();
}

std::pair<std::unique_ptr<TrainingData>, std::unique_ptr<TrainingData>>
Manager::splitData(std::unique_ptr<TrainingData> &data, float split_ratio) {
  // Shuffle the data randomly for unbiased training and validation
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data->begin(), data->end(), g);

  // Calculate the split index based on the split ratio
  size_t split_index = static_cast<size_t>(data->size() * split_ratio);

  // Split the data into training and validation sets
  auto training_data = std::make_unique<TrainingData>(
      data->begin(), data->begin() + split_index);
  auto validation_data =
      std::make_unique<TrainingData>(data->begin() + split_index, data->end());

  return std::make_pair(std::move(training_data), std::move(validation_data));
}
