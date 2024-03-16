#include "Manager.h"
#include "AppParameters.h"
#include "Common.h"
#include "RGBA.h"
#include "SimpleLogger.h"
#include <csignal>
#include <vector>

using namespace sipai;

volatile std::sig_atomic_t stopTraining = false;

void signalHandler(int signal) {
  if (signal == SIGINT) {
    std::cout << "Received interrupt signal (CTRL+C). Training will stop after "
                 "the current epoch."
              << std::endl;
    stopTraining = true;
  }
}

std::vector<RGBA> Manager::loadImage(const std::string &imagePath) {
  ImageImport imageImport;
  cv::Mat image = imageImport.importImage(imagePath);

  // Resize the image to the input neurons
  const auto &network_params = Manager::getInstance().network_params;
  cv::resize(
      image, image,
      cv::Size(network_params.input_size_x, network_params.input_size_y));

  return imageImport.convertToRGBAVector(image);
}

void Manager::run() {
  switch (app_params.run_mode) {
  case ERunMode::TrainingMonitored:
    runTrainingMonitored();
    break;
  default:
    break;
  }
}

void Manager::runTrainingMonitored() {
  // Set up signal handler
  std::signal(SIGINT, signalHandler);

  // Load training data
  trainingData trainingData = loadTrainingData();

  // Split training data into training and validation sets
  auto [trainingDataPairs, validationDataPairs] =
      splitData(trainingData, app_params.split_ratio);

  // Initialize network
  initializeNetwork();

  // Training loop
  int epoch = 0;
  float bestValidationLoss = std::numeric_limits<float>::max();
  int epochsWithoutImprovement = 0;

  while (!stopTraining &&
         shouldContinueTraining(epoch, epochsWithoutImprovement, app_params)) {
    float trainingLoss = trainOnEpoch(trainingDataPairs);
    float validationLoss = evaluateOnValidationSet(validationDataPairs);

    logTrainingProgress(epoch, trainingLoss, validationLoss);

    if (validationLoss < bestValidationLoss) {
      bestValidationLoss = validationLoss;
      epochsWithoutImprovement = 0;
    } else {
      epochsWithoutImprovement++;
    }

    epoch++;
  }
}

float Manager::trainOnEpoch(const trainingData &dataSet) {
  float epochLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : dataSet) {
    std::vector<RGBA> inputImage = loadImage(inputPath);
    std::vector<RGBA> targetImage = loadImage(targetPath);

    std::vector<RGBA> outputImage = network->forwardPropagation(inputImage);
    float loss = computeMSELoss(outputImage, targetImage);
    epochLoss += loss;

    network->backwardPropagation(targetImage);
    network->updateWeights(network_params.learning_rate);
  }
  epochLoss /= dataSet.size();
  return epochLoss;
}

float Manager::evaluateOnValidationSet(const trainingData &validationSet) {
  float validationLoss = 0.0f;
  for (const auto &[inputPath, targetPath] : validationSet) {
    std::vector<RGBA> inputImage = loadImage(inputPath);
    std::vector<RGBA> targetImage = loadImage(targetPath);

    std::vector<RGBA> outputImage = network->forwardPropagation(inputImage);
    validationLoss += computeMSELoss(outputImage, targetImage);
  }
  validationLoss /= validationSet.size();
  return validationLoss;
}

bool Manager::shouldContinueTraining(int epoch, int epochsWithoutImprovement,
                                     const AppParameters &appParams) {
  return (epochsWithoutImprovement <
          appParams.max_epochs_without_improvement) ||
         (epoch < appParams.max_epochs && appParams.max_epochs != NOMAX_EPOCHS);
}

void Manager::logTrainingProgress(int epoch, float trainingLoss,
                                  float validationLoss) {
  SimpleLogger::LOG_INFO("Epoch: ", epoch, ", Train Loss: ", trainingLoss,
                         ", Validation Loss: ", validationLoss);
}

trainingData Manager::loadTrainingData() {
  // TODO
  return {};
}

std::pair<trainingData, trainingData> Manager::splitData(trainingData data,
                                                         float split_ratio) {
  // TODO
  return {};
}

void Manager::initializeNetwork() {
  // TODO
}

/**
 * @brief Computes the mean squared error (MSE) loss between the output image
 * and the target image.
 *
 * @param outputImage The output image produced by the neural network.
 * @param targetImage The expected target image.
 *
 * @return The computed MSE loss.
 */
float Manager::computeMSELoss(const std::vector<RGBA> &outputImage,
                              const std::vector<RGBA> &targetImage) {
  if (outputImage.size() != targetImage.size()) {
    throw std::invalid_argument(
        "Output and target images must have the same size.");
  }

  float totalLoss = 0.0f;
  for (size_t i = 0; i < outputImage.size(); ++i) {
    for (size_t j = 0; j < 4; ++j) {
      float diff = outputImage[i].value[j] - targetImage[i].value[j];
      totalLoss += diff * diff;
    }
  }

  return totalLoss / (outputImage.size() * 4);
}