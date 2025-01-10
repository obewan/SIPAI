#include "RunnerTrainingOpenCVVisitor.h"
#include "AppParams.h"
#include "Common.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "TrainingDataFactory.h"
#include "exception/RunnerVisitorException.h"
#include <cstddef>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

using namespace sipai;

void RunnerTrainingOpenCVVisitor::visit() const {
  SimpleLogger::LOG_INFO(
      "Starting training monitored, press (CTRL+C) to stop at anytime...");

  auto &manager = Manager::getInstance();
  if (!manager.network) {
    throw RunnerVisitorException("No neural network. Aborting.");
  }

  const auto &appParams = manager.app_params;
  auto &learning_rate = manager.network_params.learning_rate;
  const auto &adaptive_learning_rate =
      manager.network_params.adaptive_learning_rate;
  const auto &enable_adaptive_increase =
      manager.network_params.enable_adaptive_increase;
  auto &trainingDataFactory = TrainingDataFactory::getInstance();

  const auto start{std::chrono::steady_clock::now()}; // starting timer
  SimpleLogger::getInstance().setPrecision(2);

  try {
    // Load training data
    if (appParams.verbose_debug) {
      SimpleLogger::LOG_DEBUG("Loading images data...");
    }
    trainingDataFactory.loadData();
    if (!trainingDataFactory.isLoaded() ||
        trainingDataFactory.getSize(TrainingPhase::Training) == 0) {
      throw RunnerVisitorException("No training data found. Aborting.");
    }

    // Reset the stopTraining flag
    stopTraining = false;
    stopTrainingNow = false;

    // Set up signal handler
    std::signal(SIGINT, signalHandler);

    float trainingLoss = 0.0f;
    float validationLoss = 0.0f;
    float previousTrainingLoss = 0.0f;
    float previousValidationLoss = 0.0f;
    int epoch = 0;
    int epochsWithoutImprovement = 0;
    bool hasLastEpochBeenSaved = false;

    while (!stopTraining && !stopTrainingNow &&
           shouldContinueTraining(epoch, epochsWithoutImprovement, appParams)) {

      // if Adaptive Learning Rate enabled, adapt the learning rate.
      if (adaptive_learning_rate && epoch > 1) {
        adaptLearningRate(learning_rate, validationLoss, previousValidationLoss,
                          enable_adaptive_increase);
      }

      TrainingDataFactory::getInstance().shuffle(TrainingPhase::Training);

      previousTrainingLoss = trainingLoss;
      previousValidationLoss = validationLoss;

      trainingLoss = training(epoch, TrainingPhase::Training);
      if (stopTrainingNow) {
        break;
      }

      validationLoss = training(epoch, TrainingPhase::Validation);
      if (stopTrainingNow) {
        break;
      }

      logTrainingProgress(epoch, trainingLoss, validationLoss,
                          previousTrainingLoss, previousValidationLoss);

      // check the epochs without improvement counter
      if (epoch > 0) {
        if (validationLoss < previousValidationLoss ||
            trainingLoss < previousTrainingLoss) {
          epochsWithoutImprovement = 0;
        } else {
          epochsWithoutImprovement++;
        }
      }

      hasLastEpochBeenSaved = false;
      epoch++;

      if (!appParams.no_save && !stopTrainingNow && (epoch % appParams.epoch_autosave == 0)) {
        // TODO: an option to save the best validation rate network (if not
        // saved)
        saveNetwork(hasLastEpochBeenSaved);
      }
    }

    SimpleLogger::LOG_INFO("Exiting training...");
    if (!appParams.no_save && !stopTrainingNow) {
      saveNetwork(hasLastEpochBeenSaved);
    }
    // Show elapsed time
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration elapsed_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    const auto &hms = Common::getHMSfromS(elapsed_seconds.count());
    SimpleLogger::LOG_INFO("Elapsed time: ", hms[0], "h ", hms[1], "m ", hms[2],
                           "s");

  } catch (std::exception &ex) {
    throw RunnerVisitorException(ex.what());
  }
}

float RunnerTrainingOpenCVVisitor::training(size_t epoch,
                                            TrainingPhase phase) const {

  // Initialize the total loss to 0
  float loss = 0.0f;
  size_t lossComputed = 0;
  size_t counter = 0;
  bool isLossFrequency = false;
  auto &trainingDataFactory = TrainingDataFactory::getInstance();
  trainingDataFactory.resetCounters();
  const auto &app_params = Manager::getConstInstance().app_params;

  // Compute the frequency at which the loss should be computed
  size_t lossFrequency = std::max(
      static_cast<size_t>(std::sqrt(trainingDataFactory.getSize(phase))),
      (size_t)1);

  // Loop over all images
  while (auto data = trainingDataFactory.next(phase)) {
    if (stopTrainingNow) {
      break;
    }
    counter++;
    if (app_params.verbose) {
      SimpleLogger::LOG_INFO(
          "Epoch: ", epoch + 1, ", ", Common::getTrainingPhaseStr(phase), ": ",
          "image ", counter, "/", trainingDataFactory.getSize(phase), "...");
    }

    // Check if the loss should be computed for the current image
    isLossFrequency = counter % lossFrequency == 0 ? true : false;

    // Compute the image parts loss
    float imageLoss = _training(epoch, data, phase, isLossFrequency);
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

float RunnerTrainingOpenCVVisitor::_training(size_t epoch,
                                             std::shared_ptr<Data> data,
                                             TrainingPhase phase,
                                             bool isLossFrequency) const {
  if (data->img_input.size() != data->img_target.size()) {
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
  for (size_t i = 0; i < data->img_input.size(); i++) {
    if (stopTrainingNow) {
      break;
    }

    // Get the input and target parts
    const auto &inputPart = data->img_input.at(i);
    const auto &targetPart = data->img_target.at(i);

    // Perform forward propagation
    if (manager.app_params.verbose_debug) {
      SimpleLogger::LOG_DEBUG("forward propagation part ", i + 1, "/",
                              data->img_input.size(), "...");
    }
    const auto &outputData =
        manager.network->forwardPropagation(inputPart->data);

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
    if (phase == TrainingPhase::Training) {
      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("backward propagation part ", i + 1, "/",
                                data->img_input.size(), "...");
      }
      manager.network->backwardPropagation(targetPart->data, error_min,
                                           error_max);
      if (stopTrainingNow) {
        break;
      }

      if (manager.app_params.verbose_debug) {
        SimpleLogger::LOG_DEBUG("weights update part ", i + 1, "/",
                                data->img_input.size(), "...");
      }
      manager.network->updateWeights(manager.network_params.learning_rate);
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
