#include "RunnerTrainingMonitoredVisitor.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include <csignal>
#include <exception>
#include <memory>

using namespace sipai;

volatile std::sig_atomic_t stopTraining = false;

void signalHandler(int signal) {
  if (signal == SIGINT) {
    SimpleLogger::LOG_INFO(
        "Received interrupt signal (CTRL+C). Training will stop after "
        "the current epoch.");
    stopTraining = true;
  }
}

void RunnerTrainingMonitoredVisitor::visit() const {
  SimpleLogger::LOG_INFO(
      "Starting training monitored, press (CTRL+C) to stop at anytime...");
  auto &manager = Manager::getInstance();
  const auto &appParams = Manager::getInstance().app_params;
  const auto start{std::chrono::steady_clock::now()}; // starting timer
  SimpleLogger::getInstance().setPrecision(2);

  // Load training data
  auto trainingData = manager.loadTrainingData();
  if (trainingData->empty()) {
    SimpleLogger::LOG_ERROR("No training data found. Aborting.");
    return;
  }

  try {
    // Split training data into training and validation sets
    const auto &[trainingDataPairs, validationDataPairs] =
        manager.splitData(trainingData, manager.app_params.split_ratio);

    // Reset the stopTraining flag
    stopTraining = false;

    // Set up signal handler
    std::signal(SIGINT, signalHandler);

    float previousTrainingLoss = std::numeric_limits<float>::max();
    float previousValidationLoss = std::numeric_limits<float>::max();
    int epoch = 0;
    int epochsWithoutImprovement = 0;
    bool hasLastEpochBeenSaved = false;
    while (!stopTraining &&
           shouldContinueTraining(epoch, epochsWithoutImprovement, appParams)) {
      float trainingLoss = trainOnEpoch(trainingDataPairs);
      float validationLoss = evaluateOnValidationSet(validationDataPairs);

      logTrainingProgress(epoch, trainingLoss, validationLoss);

      hasLastEpochBeenSaved = false;
      epoch++;

      if (validationLoss < previousValidationLoss ||
          trainingLoss < previousTrainingLoss) {
        epochsWithoutImprovement = 0;
      } else {
        epochsWithoutImprovement++;
      }

      previousTrainingLoss = trainingLoss;
      previousValidationLoss = validationLoss;

      if (epoch % appParams.epoch_autosave == 0) {
        saveNetwork(hasLastEpochBeenSaved);
      }
    }

    SimpleLogger::LOG_INFO("Exiting training...");
    saveNetwork(hasLastEpochBeenSaved);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};
    SimpleLogger::LOG_INFO("Elapsed time: ", elapsed_seconds.count(), "s");

  } catch (std::exception &ex) {
    SimpleLogger::LOG_ERROR("Training error: ", ex.what());
  }
}

float RunnerTrainingMonitoredVisitor::trainOnEpoch(
    const std::unique_ptr<TrainingData> &dataSet) const {
  auto &manager = Manager::getInstance();
  ImageHelper imageHelper;
  float epochLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : *dataSet) {
    size_t orig_ix;
    size_t orig_iy;
    size_t orig_tx;
    size_t orig_ty;
    std::vector<RGBA> inputImage = manager.loadImage(
        inputPath, orig_ix, orig_iy, manager.network_params.input_size_x,
        manager.network_params.input_size_y);
    std::vector<RGBA> targetImage = manager.loadImage(
        targetPath, orig_tx, orig_ty, manager.network_params.output_size_x,
        manager.network_params.output_size_y);

    std::vector<RGBA> outputImage = manager.network->forwardPropagation(
        inputImage, manager.app_params.enable_parallel);
    epochLoss += imageHelper.computeLoss(outputImage, targetImage);

    manager.network->backwardPropagation(targetImage,
                                         manager.app_params.enable_parallel);
    manager.network->updateWeights(manager.network_params.learning_rate,
                                   manager.app_params.enable_parallel);
  }
  epochLoss /= dataSet->size();
  return epochLoss;
}

float RunnerTrainingMonitoredVisitor::evaluateOnValidationSet(
    const std::unique_ptr<TrainingData> &validationSet) const {
  auto &manager = Manager::getInstance();
  ImageHelper imageHelper;
  float validationLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : *validationSet) {
    size_t orig_ix;
    size_t orig_iy;
    size_t orig_tx;
    size_t orig_ty;
    auto inputImage = manager.loadImage(inputPath, orig_ix, orig_iy,
                                        manager.network_params.input_size_x,
                                        manager.network_params.input_size_y);
    auto targetImage = manager.loadImage(targetPath, orig_tx, orig_ty,
                                         manager.network_params.output_size_x,
                                         manager.network_params.output_size_y);

    auto outputImage = manager.network->forwardPropagation(
        inputImage, manager.app_params.enable_parallel);
    validationLoss += imageHelper.computeLoss(outputImage, targetImage);
  }
  validationLoss /= validationSet->size();
  return validationLoss;
}

bool RunnerTrainingMonitoredVisitor::shouldContinueTraining(
    int epoch, int epochsWithoutImprovement, const AppParams &appParams) const {
  bool improvementCondition =
      epochsWithoutImprovement < appParams.max_epochs_without_improvement;
  bool epochCondition =
      (appParams.max_epochs == NOMAX_EPOCHS) || (epoch < appParams.max_epochs);

  return improvementCondition && epochCondition;
}

void RunnerTrainingMonitoredVisitor::logTrainingProgress(
    int epoch, float trainingLoss, float validationLoss) const {
  SimpleLogger::LOG_INFO("Epoch: ", epoch + 1,
                         ", Train Loss: ", trainingLoss * 100.0f,
                         "%, Validation Loss: ", validationLoss * 100.0f, "%");
}

void RunnerTrainingMonitoredVisitor::saveNetwork(
    bool &hasLastEpochBeenSaved) const {
  try {
    if (!hasLastEpochBeenSaved) {
      Manager::getInstance().exportNetwork();
      hasLastEpochBeenSaved = true;
    }
  } catch (std::exception &ex) {
    SimpleLogger::LOG_INFO("Saving the neural network error: ", ex.what());
  }
}