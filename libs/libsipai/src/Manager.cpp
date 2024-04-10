#include "Manager.h"
#include "AppParams.h"
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

  SimpleLogger::LOG_INFO(
      "Parameters: ", "\nmode: ", getRunModeStr(app_params.run_mode),
      "\nauto-save every ", app_params.epoch_autosave, " epochs",
      "\nauto-exit after ", app_params.max_epochs_without_improvement,
      " epochs without improvement",
      app_params.max_epochs == NO_MAX_EPOCHS
          ? "\nno maximum epochs"
          : "\nauto-exit after a maximum of " +
                std::to_string(app_params.max_epochs) + " epochs",
      "\ntraining/validation ratio: ", app_params.training_split_ratio,
      "\nlearning rate: ", network_params.learning_rate,
      "\ninput layer size: ", network_params.input_size_x, "x",
      network_params.input_size_y,
      "\nhidden layer size: ", network_params.hidden_size_x, "x",
      network_params.hidden_size_y,
      "\noutput layer size: ", network_params.output_size_x, "x",
      network_params.output_size_y,
      "\nhidden layers: ", network_params.hiddens_count,
      "\nhidden activation function: ",
      getActivationStr(network_params.hidden_activation_function),
      "\nhidden activation alpha: ", network_params.hidden_activation_alpha,
      "\noutput activation function: ",
      getActivationStr(network_params.output_activation_function),
      "\noutput activation alpha: ", network_params.output_activation_alpha,
      "\nimage split: ", app_params.image_split,
      "\noutput scale: ", app_params.output_scale,
      "\nimages bulk loading: ", app_params.bulk_loading ? "true" : "false",
      "\nparallelism enabled: ", app_params.enable_parallel ? "true" : "false");

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
