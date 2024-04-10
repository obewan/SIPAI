#include "RunnerTrainingMonitoredVisitor.h"
#include "AppParams.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
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
  const auto &appParams = manager.app_params;

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
        manager.splitData(trainingData, appParams.training_split_ratio);

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
    training_images_ =
        std::make_unique<std::vector<std::pair<ImageParts, ImageParts>>>(
            loadBulkImages(trainingSet, "Training:"));
  }
  float epochLoss = manager.app_params.bulk_loading
                        ? computeLoss(*training_images_, true)
                        : computeLoss(*trainingSet, true);
  return epochLoss;
}

float RunnerTrainingMonitoredVisitor::evaluateOnValidationSet(
    const std::unique_ptr<TrainingData> &validationSet) const {
  auto &manager = Manager::getInstance();
  if (manager.app_params.bulk_loading && !validation_images_) {
    validation_images_ =
        std::make_unique<std::vector<std::pair<ImageParts, ImageParts>>>(
            loadBulkImages(validationSet, "Validation:"));
  }
  float validationLoss = manager.app_params.bulk_loading
                             ? computeLoss(*validation_images_, false)
                             : computeLoss(*validationSet, false);
  return validationLoss;
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

std::pair<ImageParts, ImageParts> RunnerTrainingMonitoredVisitor::loadImages(
    const std::string &inputPath, const std::string &targetPath) const {
  const auto &network_params = Manager::getInstance().network_params;
  const auto &app_params = Manager::getInstance().app_params;

  // Load and split the input image
  const auto &inputImageParts = imageHelper_.loadImage(
      inputPath, app_params.image_split, network_params.input_size_x,
      network_params.input_size_y);

  // Load and split the target image
  const auto &targetImageParts = imageHelper_.loadImage(
      targetPath, app_params.image_split, network_params.output_size_x,
      network_params.output_size_y);

  // Check if the number of parts is the same for both images
  if (inputImageParts.size() != targetImageParts.size()) {
    throw RunnerVisitorException(
        "Mismatch in number of image parts: expected " +
        std::to_string(inputImageParts.size()) + ", got " +
        std::to_string(targetImageParts.size()));
  }
  return std::make_pair(inputImageParts, targetImageParts);
}

std::vector<std::pair<ImageParts, ImageParts>>
RunnerTrainingMonitoredVisitor::loadBulkImages(
    const std::unique_ptr<TrainingData> &dataSet, std::string logPrefix) const {
  std::vector<std::pair<ImageParts, ImageParts>> images;
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
          std::pair<ImageParts, ImageParts> pair =
              loadImages(inputPath, targetPath);

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

  SimpleLogger::LOG_INFO(logPrefix + " images loaded...");

  return images;
}

float RunnerTrainingMonitoredVisitor::computeLoss(
    const ImageParts &inputImage, const ImageParts &targetImage,
    bool withBackwardAndUpdateWeights, bool isLossFrequency) const {
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
        inputPart.data, manager.app_params.enable_parallel);

    // If the loss should be computed for the current image, compute the loss
    // for the current part
    if (isLossFrequency) {
      partsLoss += imageHelper_.computeLoss(outputData, targetPart.data);
      partsLossComputed++;
    }

    // If backward propagation and weight update should be performed, perform
    // them
    if (withBackwardAndUpdateWeights) {
      manager.network->backwardPropagation(targetPart.data,
                                           manager.app_params.enable_parallel);
      manager.network->updateWeights(manager.network_params.learning_rate,
                                     manager.app_params.enable_parallel);
    }
  }

  return (partsLoss / static_cast<float>(partsLossComputed));
}

float RunnerTrainingMonitoredVisitor::computeLoss(
    const std::vector<std::pair<ImageParts, ImageParts>> &images,
    bool withBackwardAndUpdateWeights) const {
  // Initialize the total loss to 0
  float loss = 0.0f;
  size_t lossComputed = 0;
  size_t counter = 0;
  bool isLossFrequency = false;

  // Compute the frequency at which the loss should be computed
  size_t lossFrequency =
      std::max(static_cast<size_t>(std::sqrt(images.size())), (size_t)1);

  // Loop over all images
  for (const auto &[inputImageParts, targetImageParts] : images) {
    counter++;

    // Check if the loss should be computed for the current image
    isLossFrequency = counter % lossFrequency == 0 ? true : false;

    // Compute the image parts loss
    float imageLoss =
        computeLoss(inputImageParts, targetImageParts,
                    withBackwardAndUpdateWeights, isLossFrequency);

    // If the loss was computed for the current image, add the average loss for
    // the current image to the total loss
    if (isLossFrequency) {
      loss += imageLoss;
      lossComputed++;
    }
  }

  // Return the average loss over all images for which the loss was computed
  return (loss / static_cast<float>(lossComputed));
}

float RunnerTrainingMonitoredVisitor::computeLoss(
    const TrainingData &dataSet, bool withBackwardAndUpdateWeights) const {
  // Initialize the total loss to 0
  float loss = 0.0f;
  size_t lossComputed = 0;
  size_t counter = 0;
  bool isLossFrequency = false;

  // Compute the frequency at which the loss should be computed
  size_t lossFrequency =
      std::max(static_cast<size_t>(std::sqrt(dataSet.size())), (size_t)1);

  // Loop over all images
  for (const auto &[inputPath, targetPath] : dataSet) {
    // Load the image parts
    const auto &[inputImageParts, targetImageParts] =
        loadImages(inputPath, targetPath);
    counter++;

    // Check if the loss should be computed for the current image
    isLossFrequency = counter % lossFrequency == 0 ? true : false;

    // Compute the image parts loss
    float imageLoss =
        computeLoss(inputImageParts, targetImageParts,
                    withBackwardAndUpdateWeights, isLossFrequency);

    // If the loss was computed for the current image, add the average loss for
    // the current image to the total loss
    if (isLossFrequency) {
      loss += imageLoss;
      lossComputed++;
    }

    // Unload images
  }
  return (loss / static_cast<float>(lossComputed));
}