#include "RunnerTrainingVisitor.h"
#include "Manager.h"
#include "SimpleLogger.h"

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

bool RunnerTrainingVisitor::shouldContinueTraining(
    int epoch, size_t epochsWithoutImprovement,
    const AppParams &appParams) const {
  bool improvementCondition =
      epochsWithoutImprovement < appParams.max_epochs_without_improvement;
  bool epochCondition =
      (appParams.max_epochs == NO_MAX_EPOCHS) || (epoch < (int)appParams.max_epochs);

  return improvementCondition && epochCondition;
}

void RunnerTrainingVisitor::adaptLearningRate(
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

void RunnerTrainingVisitor::logTrainingProgress(
    const int &epoch, const float &trainingLoss, const float &validationLoss,
    const float &previousTrainingLoss,
    const float &previousValidationLoss) const {
  std::stringstream delta;
  if (epoch > 0) {
    float dtl = trainingLoss - previousTrainingLoss;
    float dvl = validationLoss - previousValidationLoss;
    delta.precision(2);
    delta << std::fixed << " [" << (dtl > 0 ? "+" : "") << dtl * 100.0f << "%";
    delta << std::fixed << "," << (dvl > 0 ? "+" : "") << dvl * 100.0f << "%]";
  }
  SimpleLogger::LOG_INFO(
      "Epoch: ", epoch + 1, ", Train Loss: ", trainingLoss * 100.0f,
      "%, Validation Loss: ", validationLoss * 100.0f, "%", delta.str());
}

void RunnerTrainingVisitor::saveNetwork(bool &hasLastEpochBeenSaved, std::function<void(int)> progressCallback) const {
  std::scoped_lock<std::mutex> lock(threadMutex_);
  try {
    if (!hasLastEpochBeenSaved) {
      Manager::getInstance().exportNetwork(progressCallback);
      hasLastEpochBeenSaved = true;
    }
  } catch (std::exception &ex) {
    SimpleLogger::LOG_INFO("Saving the neural network error: ", ex.what());
  }
}