#include "Manager.h"
#include "AppParams.h"
#include "Common.h"
#include "NeuralNetwork.h"
#include "SimpleLogger.h"
#include "VulkanController.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <numeric>

using namespace sipai;

std::unique_ptr<Manager> Manager::instance_ = nullptr;

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
      "\nadaptive learning rate: ",
      network_params.adaptive_learning_rate ? "true" : "false",
      "\nadaptive learning rate increase: ",
      network_params.enable_adaptive_increase ? "true" : "false",
      "\nadaptive learning rate factor: ",
      network_params.adaptive_learning_rate_factor,
      "\ntraining error min: ", network_params.error_min,
      "\ntraining error max: ", network_params.error_max,
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
      "\ninput reduce factor: ", app_params.training_reduce_factor,
      "\noutput scale: ", app_params.output_scale,
      "\nimages random loading: ", app_params.random_loading ? "true" : "false",
      "\nimages bulk loading: ", app_params.bulk_loading ? "true" : "false",
      "\npadding enabled: ", app_params.enable_padding ? "true" : "false",
      "\nvulkan enabled: ", app_params.enable_vulkan ? "true" : "false",
      "\nparallelism enabled: ", app_params.enable_parallel ? "true" : "false",
      "\nverbose logs enabled: ", app_params.verbose ? "true" : "false",
      "\ndebug logs enabled: ", app_params.verbose_debug ? "true" : "false");

  if (app_params.enable_vulkan) {
    SimpleLogger::LOG_INFO("Enabling Vulkan...");
    try {
      VulkanController::getInstance().initialize();
    } catch (std::exception &ex) {
      SimpleLogger::LOG_ERROR("Enabling Vulkan error: ", ex.what());
      app_params.enable_vulkan = false;
      SimpleLogger::LOG_INFO("Vulkan GPU acceleration disabled.");
    }
  }

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
