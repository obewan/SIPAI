#include "RunnerTrainingVulkanVisitor.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "TrainingDataFactory.h"
#include "VulkanController.h"
#include "exception/RunnerVisitorException.h"
#include <cstddef>
#include <memory>

using namespace sipai;

void RunnerTrainingVulkanVisitor::visit() const {
  SimpleLogger::LOG_INFO("Starting training monitored (Vulkan), press (CTRL+C) "
                         "to stop at anytime...");

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

    float previousTrainingLoss = std::numeric_limits<float>::max();
    float previousValidationLoss = std::numeric_limits<float>::max();
    int epoch = 0;
    int epochsWithoutImprovement = 0;
    bool hasLastEpochBeenSaved = false;
    while (!stopTraining && !stopTrainingNow &&
           shouldContinueTraining(epoch, epochsWithoutImprovement, appParams)) {
      TrainingDataFactory::getInstance().shuffle(TrainingPhase::Training);

      float trainingLoss = computeLoss(epoch, TrainingPhase::Training);
      if (stopTrainingNow) {
        break;
      }

      float validationLoss = computeLoss(epoch, TrainingPhase::Validation);
      if (stopTrainingNow) {
        break;
      }

      logTrainingProgress(epoch, trainingLoss, validationLoss,
                          previousTrainingLoss, previousValidationLoss);

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
        // TODO: an option to save the best validation rate network (if not
        // saved)
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
    const auto &hms = Common::getHMSfromS(elapsed_seconds.count());
    SimpleLogger::LOG_INFO("Elapsed time: ", hms[0], "h ", hms[1], "m ", hms[2],
                           "s");

  } catch (std::exception &ex) {
    throw RunnerVisitorException(ex.what());
  }
}

float RunnerTrainingVulkanVisitor::computeLoss(size_t epoch,
                                               TrainingPhase phase) const {

  // Initialize the total loss to 0
  float loss = 0.0f;
  size_t lossComputed = 0;
  size_t counter = 0;
  bool isLossFrequency = false;
  auto &trainingDataFactory = TrainingDataFactory::getInstance();
  trainingDataFactory.resetCounters();
  const auto &app_params = Manager::getConstInstance().app_params;

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
    if (data->img_input.size() == 0) {
      continue;
    }
    float imageLoss = 0;
    for (size_t i = 0; i < data->img_input.size(); i++) {
      if (stopTrainingNow) {
        break;
      }

      // Get the input and target parts
      const auto &inputPart = data->img_input.at(i);
      const auto &targetPart = data->img_target.at(i);
      imageLoss += VulkanController::getInstance().trainingMonitored(
          inputPart, targetPart, phase);
    }

    loss += (imageLoss / static_cast<float>(data->img_input.size()));
    lossComputed++;
  }

  if (lossComputed == 0) {
    return 0;
  }
  return (loss / static_cast<float>(lossComputed));
}