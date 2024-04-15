#include "RunnerTrainingMonitoredVisitor.h"
#include "AppParams.h"
#include "Common.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "TrainingDataFactory.h"
#include "exception/RunnerVisitorException.h"
#include <csignal>
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <utility>

using namespace sipai;

volatile std::sig_atomic_t stopTraining = false;

void signalHandler(int signal) {
  if (signal == SIGINT) {
    if (!stopTraining) {
      SimpleLogger::LOG_INFO(
          "Received interrupt signal (CTRL+C). Training will stop after "
          "the current epoch. Press another time on (CTRL+C) to force exit "
          "immediately without saving.");
      stopTraining = true;
    } else {
      SimpleLogger::LOG_INFO(
          "Received another interrupt signal (CTRL+C). "
          "Forcing quitting immedialty without saving progress.");
      std::exit(EXIT_SUCCESS); // Terminate the program immediately
    }
  }
}

void RunnerTrainingMonitoredVisitor::visit() const {
  SimpleLogger::LOG_INFO(
      "Starting training monitored, press (CTRL+C) to stop at anytime...");

  auto &manager = Manager::getInstance();
  const auto &appParams = manager.app_params;

  auto &trainingDataFactory = TrainingDataFactory::getInstance();

  const auto start{std::chrono::steady_clock::now()}; // starting timer
  SimpleLogger::getInstance().setPrecision(2);

  // Load training data
  trainingDataFactory.loadData();
  if (!trainingDataFactory.isLoaded() ||
      trainingDataFactory.trainingSize() == 0) {
    SimpleLogger::LOG_ERROR("No training data found. Aborting.");
    return;
  }

  try {
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
      float trainingLoss = computeLoss(epoch, true);
      float validationLoss = computeLoss(epoch, false);

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

    // Show elapsed time
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration elapsed_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    const auto &hms = getHMSfromS(elapsed_seconds.count());
    SimpleLogger::LOG_INFO("Elapsed time: ", hms[0], "h ", hms[1], "m ", hms[2],
                           "s");

  } catch (std::exception &ex) {
    SimpleLogger::LOG_ERROR("Training error: ", ex.what());
  }
}

bool RunnerTrainingMonitoredVisitor::shouldContinueTraining(
    int epoch, size_t epochsWithoutImprovement,
    const AppParams &appParams) const {
  bool improvementCondition =
      epochsWithoutImprovement < appParams.max_epochs_without_improvement;
  bool epochCondition =
      (appParams.max_epochs == NO_MAX_EPOCHS) || (epoch < appParams.max_epochs);

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

float RunnerTrainingMonitoredVisitor::computeLoss(size_t epoch,
                                                  const ImageParts &inputImage,
                                                  const ImageParts &targetImage,
                                                  bool isTraining,
                                                  bool isLossFrequency) const {
  auto &manager = Manager::getInstance();
  if (inputImage.size() != targetImage.size()) {
    throw ImageHelperException(
        "internal exception: input and target parts have different sizes.");
  }

  // Initialize the loss for the current image to 0
  float partsLoss = 0.0f;

  // Initialize a counter to keep track of the number of parts for which the
  // loss is computed
  size_t partsLossComputed = 0;

  // Loop over all parts of the current image
  for (size_t i = 0; i < inputImage.size(); i++) {
    // Get the input and target parts
    const auto &inputPart = inputImage.at(i);
    const auto &targetPart = targetImage.at(i);

    // Perform forward propagation
    const auto &outputData = manager.network->forwardPropagation(
        inputPart->data, manager.app_params.enable_parallel);

    // If the loss should be computed for the current image, compute the loss
    // for the current part
    if (isLossFrequency) {
      partsLoss += imageHelper_.computeLoss(outputData, targetPart->data);
      partsLossComputed++;
    }

    // If backward propagation and weight update should be performed, perform
    // them
    if (isTraining) {
      manager.network->backwardPropagation(targetPart->data,
                                           manager.app_params.enable_parallel);
      manager.network->updateWeights(manager.network_params.learning_rate,
                                     manager.app_params.enable_parallel);
    }
  }

  if (partsLossComputed == 0) {
    return 0;
  }
  return (partsLoss / static_cast<float>(partsLossComputed));
}

float RunnerTrainingMonitoredVisitor::computeLoss(size_t epoch,
                                                  bool isTraining) const {

  // Initialize the total loss to 0
  float loss = 0.0f;
  size_t lossComputed = 0;
  size_t counter = 0;
  bool isLossFrequency = false;
  auto &trainingDataFactory = TrainingDataFactory::getInstance();
  trainingDataFactory.resetCounters();
  const auto &app_params = Manager::getConstInstance().app_params;

  // Compute the frequency at which the loss should be computed
  size_t lossFrequency =
      std::max(static_cast<size_t>(std::sqrt(
                   isTraining ? trainingDataFactory.trainingSize()
                              : trainingDataFactory.validationSize())),
               (size_t)1);

  // Loop over all images
  while (auto imagePartsPair = isTraining
                                   ? trainingDataFactory.nextTraining()
                                   : trainingDataFactory.nextValidation()) {
    counter++;
    if (app_params.verbose) {
      SimpleLogger::LOG_INFO("Epoch: ", epoch + 1,
                             isTraining ? ", training: " : ", validation: ",
                             "image ", counter, "/",
                             isTraining ? trainingDataFactory.trainingSize()
                                        : trainingDataFactory.validationSize(),
                             "...");
    }

    // Check if the loss should be computed for the current image
    isLossFrequency = counter % lossFrequency == 0 ? true : false;

    // Compute the image parts loss
    const auto &[inputImageParts, targetImageParts] = *imagePartsPair;
    float imageLoss = computeLoss(epoch, inputImageParts, targetImageParts,
                                  isTraining, isLossFrequency);

    // If the loss was computed for the current image, add the average loss for
    // the current image to the total loss
    if (isLossFrequency) {
      loss += imageLoss;
      lossComputed++;
    }
  }

  // Return the average loss over all images for which the loss was computed
  if (lossComputed == 0) {
    return 0;
  }
  return (loss / static_cast<float>(lossComputed));
}
