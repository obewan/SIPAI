#include "TrainingMonitoredVisitor.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include <csignal>
#include <exception>

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

void TrainingMonitoredVisitor::visit(const TrainingData &dataSet,
                                     const TrainingData &validationSet) const {
  // Reset the stopTraining flag
  stopTraining = false;

  // Set up signal handler
  std::signal(SIGINT, signalHandler);

  float bestValidationLoss = std::numeric_limits<float>::max();
  int epoch = 0;
  int epochsWithoutImprovement = 0;
  const auto &appParams = Manager::getInstance().app_params;

  while (!stopTraining &&
         shouldContinueTraining(epoch, epochsWithoutImprovement, appParams)) {
    float trainingLoss = trainOnEpoch(dataSet);
    float validationLoss = evaluateOnValidationSet(validationSet);

    logTrainingProgress(epoch, trainingLoss, validationLoss);

    if (validationLoss < bestValidationLoss) {
      bestValidationLoss = validationLoss;
      epochsWithoutImprovement = 0;
    } else {
      epochsWithoutImprovement++;
    }

    epoch++;
  }

  SimpleLogger::LOG_INFO("Exiting training, saving the neural network...");
  try {
    Manager::getInstance().exportNetwork();
  } catch (std::exception &ex) {
    SimpleLogger::LOG_INFO("Saving the neural network error: ", ex.what());
  }
}

float TrainingMonitoredVisitor::trainOnEpoch(
    const TrainingData &dataSet) const {
  auto &manager = Manager::getInstance();
  float epochLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : dataSet) {
    std::vector<RGBA> inputImage = manager.loadImage(inputPath);
    std::vector<RGBA> targetImage = manager.loadImage(targetPath);

    std::vector<RGBA> outputImage =
        manager.network->forwardPropagation(inputImage);
    float loss = manager.computeMSELoss(outputImage, targetImage);
    epochLoss += loss;

    manager.network->backwardPropagation(targetImage);
    manager.network->updateWeights(manager.network_params.learning_rate);
  }
  epochLoss /= dataSet.size();
  return epochLoss;
}

float TrainingMonitoredVisitor::evaluateOnValidationSet(
    const TrainingData &validationSet) const {
  auto &manager = Manager::getInstance();
  float validationLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : validationSet) {
    std::vector<RGBA> inputImage = manager.loadImage(inputPath);
    std::vector<RGBA> targetImage = manager.loadImage(targetPath);

    std::vector<RGBA> outputImage =
        manager.network->forwardPropagation(inputImage);
    validationLoss += manager.computeMSELoss(outputImage, targetImage);
  }
  validationLoss /= validationSet.size();
  return validationLoss;
}

bool TrainingMonitoredVisitor::shouldContinueTraining(
    int epoch, int epochsWithoutImprovement, const AppParams &appParams) const {
  return (epochsWithoutImprovement <
          appParams.max_epochs_without_improvement) ||
         (epoch < appParams.max_epochs && appParams.max_epochs != NOMAX_EPOCHS);
}

void TrainingMonitoredVisitor::logTrainingProgress(int epoch,
                                                   float trainingLoss,
                                                   float validationLoss) const {
  SimpleLogger::LOG_INFO("Epoch: ", epoch, ", Train Loss: ", trainingLoss,
                         ", Validation Loss: ", validationLoss);
}