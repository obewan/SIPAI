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
volatile std::sig_atomic_t stopTrainingNow = false;

void signalHandler(int signal) {
  if (signal == SIGINT) {
    if (!stopTraining) {
      SimpleLogger::LOG_INFO(
          "Received interrupt signal (CTRL+C). Training will stop after "
          "the current epoch. Press another time on (CTRL+C) to force exit "
          "immediately without saving.");
      stopTraining = true;
    } else {
      SimpleLogger::LOG_INFO("Received another interrupt signal (CTRL+C). "
                             "Forcing quitting immedialty without saving "
                             "progress. Please wait for cleaning...");
      stopTrainingNow = true;
    }
  }
}

void RunnerTrainingMonitoredVisitor::visit() const {
  SimpleLogger::LOG_INFO(
      "Starting training monitored, press (CTRL+C) to stop at anytime...");

  auto &manager = Manager::getInstance();
  const auto &appParams = manager.app_params;
  auto &learning_rate = manager.network_params.learning_rate;
  const auto &adaptive_learning_rate =
      manager.network_params.adaptive_learning_rate;
  const auto &enable_adaptive_increase =
      manager.network_params.enable_adaptive_increase;
  auto &trainingDataFactory = TrainingDataFactory::getInstance();

  const auto start{std::chrono::steady_clock::now()}; // starting timer
  SimpleLogger::getInstance().setPrecision(2);

  // Load training data
  if (appParams.verbose_debug) {
    SimpleLogger::LOG_DEBUG("Loading images data...");
  }
  trainingDataFactory.loadData();
  if (!trainingDataFactory.isLoaded() ||
      trainingDataFactory.trainingSize() == 0) {
    SimpleLogger::LOG_ERROR("No training data found. Aborting.");
    return;
  }

  try {
    // Reset the stopTraining flag
    stopTraining = false;
    stopTrainingNow = false;

    // Set up signal handler
    std::signal(SIGINT, signalHandler);

    float previousTrainingLoss = std::numeric_limits<float>::max();
    float previousValidationLoss = std::numeric_limits<float>::max();
    int epoch = 0;
    int epochsWithoutImprovement = 0;
    bool hasLastEpochBeenSaved = false;
    while (!stopTraining && !stopTrainingNow &&
           shouldContinueTraining(epoch, epochsWithoutImprovement, appParams)) {
      float trainingLoss = computeLoss(epoch, true);
      if (stopTrainingNow) {
        break;
      }

      float validationLoss = computeLoss(epoch, false);
      if (stopTrainingNow) {
        break;
      }

      logTrainingProgress(epoch, trainingLoss, validationLoss);

      hasLastEpochBeenSaved = false;
      epoch++;

      // if Adaptive Learning Rate enabled, adapt the learning rate.
      if (adaptive_learning_rate && epoch > 1) {
        adaptLearningRate(learning_rate, validationLoss, previousValidationLoss,
                          enable_adaptive_increase);
      }

      // check the epochs without improvement counter
      if (validationLoss < previousValidationLoss ||
          trainingLoss < previousTrainingLoss) {
        epochsWithoutImprovement = 0;
      } else {
        epochsWithoutImprovement++;
      }

      previousTrainingLoss = trainingLoss;
      previousValidationLoss = validationLoss;

      if (!stopTrainingNow && (epoch % appParams.epoch_autosave == 0)) {
        saveNetwork(hasLastEpochBeenSaved);
      }
    }

    SimpleLogger::LOG_INFO("Exiting training...");
    if (!stopTrainingNow) {
      saveNetwork(hasLastEpochBeenSaved);
    }
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

void RunnerTrainingMonitoredVisitor::adaptLearningRate(
    float &learningRate, const float &validationLoss,
    const float &previousValidationLoss,
    const bool &enable_adaptive_increase) const {
  std::scoped_lock<std::mutex> lock(threadMutex_);

  const auto &manager = Manager::getConstInstance();
  const auto &appParams = manager.app_params;
  const auto &learning_rate_min = appParams.learning_rate_min;
  const auto &learning_rate_max = appParams.learning_rate_max;
  const auto &learning_rate_adaptive_factor =
      manager.network_params.adaptive_learning_rate_factor;

  const float previous_learning_rate = learningRate;
  const float increase_slower_factor = 1.5f;

  if (validationLoss >= previousValidationLoss &&
      learningRate > learning_rate_min) {
    // this will decrease learningRate (0.001 * 0.5 = 0.0005)
    learningRate *= learning_rate_adaptive_factor;
  } else if (enable_adaptive_increase &&
             validationLoss < previousValidationLoss &&
             learningRate < learning_rate_max) {
    // this will increase learningRate but slower (0.001 / (0.5 * 1.5) = 0.0013)
    learningRate /= (learning_rate_adaptive_factor * increase_slower_factor);
  }
  learningRate = std::clamp(learningRate, learning_rate_min, learning_rate_max);

  if (appParams.verbose && learningRate != previous_learning_rate) {
    const auto current_precision = SimpleLogger::getInstance().getPrecision();
    SimpleLogger::getInstance()
        .setPrecision(6)
        .info("Learning rate ", previous_learning_rate, " adjusted to ",
              learningRate)
        .setPrecision(current_precision);
  }
}

void RunnerTrainingMonitoredVisitor::saveNetwork(
    bool &hasLastEpochBeenSaved) const {
  std::scoped_lock<std::mutex> lock(threadMutex_);
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
  if (inputImage.size() != targetImage.size()) {
    throw ImageHelperException(
        "internal exception: input and target parts have different sizes.");
  }

  auto &manager = Manager::getInstance();
  const auto &error_min = manager.network_params.error_min;
  const auto &error_max = manager.network_params.error_max;

  // Initialize the loss for the current image to 0
  float partsLoss = 0.0f;

  // Initialize a counter to keep track of the number of parts for which the
  // loss is computed
  size_t partsLossComputed = 0;

  // Loop over all parts of the current image
  for (size_t i = 0; i < inputImage.size(); i++) {
    if (stopTrainingNow) {
      break;
    }

    // Get the input and target parts
    const auto &inputPart = inputImage.at(i);
    const auto &targetPart = targetImage.at(i);

    // Perform forward propagation
    if (manager.app_params.verbose_debug) {
      SimpleLogger::LOG_DEBUG("forward propagation part ", i + 1, "/",
                              inputImage.size(), "...");
    }
    const auto &outputData = manager.network->forwardPropagation(
        inputPart->data, manager.app_params.enable_vulkan,
        manager.app_params.enable_parallel);

    if (stopTrainingNow) {
      break;
    }

    // If the loss should be computed for the current image, compute the loss
    // for the current part
    if (isLossFrequency) {
      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("loss computation...");
      }
      float partLoss = imageHelper_.computeLoss(outputData, targetPart->data);
      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("part loss: ", partLoss * 100.0f, "%");
      }
      partsLoss += partLoss;
      partsLossComputed++;
    }
    if (stopTrainingNow) {
      break;
    }

    // If backward propagation and weight update should be performed, perform
    // them
    if (isTraining) {
      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("backward propagation part ", i + 1, "/",
                                inputImage.size(), "...");
      }
      manager.network->backwardPropagation(targetPart->data, error_min,
                                           error_max,
                                           manager.app_params.enable_parallel);
      if (stopTrainingNow) {
        break;
      }

      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("weights update part ", i + 1, "/",
                                inputImage.size(), "...");
      }
      manager.network->updateWeights(manager.network_params.learning_rate,
                                     manager.app_params.enable_parallel);
      if (stopTrainingNow) {
        break;
      }
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
    if (stopTrainingNow) {
      break;
    }
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
    if (stopTrainingNow) {
      break;
    }

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
