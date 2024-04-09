#include "RunnerTrainingMonitoredVisitor.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/RunnerVisitorException.h"
#include <csignal>
#include <exception>
#include <memory>
#include <utility>

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
    const std::unique_ptr<TrainingData> &trainingSet) const {
  auto &manager = Manager::getInstance();
  if (manager.app_params.bulk_loading && !training_images_) {
    training_images_ = std::make_unique<std::vector<std::pair<Image, Image>>>(
        loadBulkImages(trainingSet, "Training:"));
  }
  float epochLoss = manager.app_params.bulk_loading
                        ? computeLoss(*training_images_, true)
                        : computeLoss(*trainingSet, true);
  epochLoss /= trainingSet->size();
  return epochLoss;
}

float RunnerTrainingMonitoredVisitor::evaluateOnValidationSet(
    const std::unique_ptr<TrainingData> &validationSet) const {
  auto &manager = Manager::getInstance();
  if (manager.app_params.bulk_loading && !validation_images_) {
    validation_images_ = std::make_unique<std::vector<std::pair<Image, Image>>>(
        loadBulkImages(validationSet, "Validation:"));
  }
  float validationLoss = manager.app_params.bulk_loading
                             ? computeLoss(*validation_images_, false)
                             : computeLoss(*validationSet, false);
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

std::pair<Image, Image> RunnerTrainingMonitoredVisitor::loadImages(
    const std::string &inputPath, const std::string &targetPath) const {
  const auto &network_params = Manager::getInstance().network_params;
  const auto &inputImage = imageHelper_.loadImage(
      inputPath, network_params.input_size_x, network_params.input_size_y);
  const auto &targetImage = imageHelper_.loadImage(
      targetPath, network_params.output_size_x, network_params.output_size_y);
  return std::make_pair(inputImage, targetImage);
}

std::vector<std::pair<Image, Image>>
RunnerTrainingMonitoredVisitor::loadBulkImages(
    const std::unique_ptr<TrainingData> &dataSet, std::string logPrefix) const {
  std::vector<std::pair<Image, Image>> images;
  SimpleLogger::LOG_INFO(logPrefix + " loading images... (bulk loading)");

  std::mutex images_mutex;
  std::atomic<size_t> current_index = 0; // atomicity between threads
  size_t num_threads =
      std::min((size_t)std::thread::hardware_concurrency(), dataSet->size());

  std::vector<std::jthread> threads;
  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([this, &dataSet, &images, &images_mutex,
                          &current_index, logPrefix]() {
      try {
        while (true) {
          size_t index = current_index.fetch_add(1);
          if (index >= dataSet->size()) {
            break;
          }

          auto it = std::next(dataSet->begin(), index);
          const auto &[inputPath, targetPath] = *it;
          std::pair<Image, Image> pair = loadImages(inputPath, targetPath);

          {
            std::lock_guard<std::mutex> lock(images_mutex);
            images.emplace_back(std::move(pair));
          }
        }
      } catch (const std::bad_alloc &e) {
        throw RunnerVisitorException(logPrefix +
                                     " loading images error: out of memory.");
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  return images;
}

float RunnerTrainingMonitoredVisitor::computeLoss(
    const std::vector<std::pair<Image, Image>> &images,
    bool withBackwardAndUpdateWeights) const {
  auto &manager = Manager::getInstance();
  float loss = 0.0f;
  for (const auto &[inputImage, targetImage] : images) {
    const auto &outputData = manager.network->forwardPropagation(
        inputImage.data, manager.app_params.enable_parallel);
    loss += imageHelper_.computeLoss(outputData, targetImage.data);
    if (withBackwardAndUpdateWeights) {
      manager.network->backwardPropagation(targetImage.data,
                                           manager.app_params.enable_parallel);
      manager.network->updateWeights(manager.network_params.learning_rate,
                                     manager.app_params.enable_parallel);
    }
  }
  return loss;
}

float RunnerTrainingMonitoredVisitor::computeLoss(
    const TrainingData &dataSet, bool withBackwardAndUpdateWeights) const {
  auto &manager = Manager::getInstance();
  float loss = 0.0f;
  for (const auto &[inputPath, targetPath] : dataSet) {
    const auto &[inputImage, targetImage] = loadImages(inputPath, targetPath);
    const auto &outputData = manager.network->forwardPropagation(
        inputImage.data, manager.app_params.enable_parallel);
    loss += imageHelper_.computeLoss(outputData, targetImage.data);
    if (withBackwardAndUpdateWeights) {
      manager.network->backwardPropagation(targetImage.data,
                                           manager.app_params.enable_parallel);
      manager.network->updateWeights(manager.network_params.learning_rate,
                                     manager.app_params.enable_parallel);
    }
  }
  return loss;
}