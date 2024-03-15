#include "Manager.h"
#include "Common.h"
#include "RGBA.h"
#include <vector>

using namespace sipai;

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
  // Load training data
  trainingData trainingData = loadTrainingData();

  // Split training data into training and validation sets
  auto [trainData, valData] = splitData(trainingData, 0.8);

  // Initialize network
  initializeNetwork();

  // Training loop
  int epoch = 0;
  float bestValLoss = std::numeric_limits<float>::max();
  while (true) {
    float trainLoss = 0.0f;
    for (const auto &[inputPath, targetPath] : trainData) {
      // Load input and target images
      std::vector<RGBA> inputImage = loadImage(inputPath);
      std::vector<RGBA> targetImage = loadImage(targetPath);

      // Forward propagation
      std::vector<RGBA> outputImage = network->forwardPropagation(inputImage);

      // Compute loss
      float loss = computeLoss(outputImage, targetImage);
      trainLoss += loss;

      // Backward propagation
      network->backwardPropagation(targetImage);

      // Update weights
      network->updateWeights(network_params.learning_rate);
    }
    trainLoss /= trainData.size();

    // Evaluate on validation set
    float valLoss = 0.0f;
    for (const auto &[inputPath, targetPath] : valData) {
      std::vector<RGBA> inputImage = loadImage(inputPath);
      std::vector<RGBA> targetImage = loadImage(targetPath);

      std::vector<RGBA> outputImage = network->forwardPropagation(inputImage);
      valLoss += computeLoss(outputImage, targetImage);
    }
    valLoss /= valData.size();

    // Log training progress
    std::cout << "Epoch: " << epoch << ", Train Loss: " << trainLoss
              << ", Validation Loss: " << valLoss << std::endl;

    // Early stopping
    if (valLoss < bestValLoss) {
      bestValLoss = valLoss;
    } else {
      // Stop training if validation loss doesn't improve for a certain
      // number of epochs
      break;
    }

    epoch++;
  }
}

trainingData Manager::loadTrainingData() {
  // TODO
  return {};
}

std::pair<trainingData, trainingData> Manager::splitData(trainingData data,
                                                         float split_offset) {
  // TODO
  return {};
}

void Manager::initializeNetwork() {
  // TODO
}

float Manager::computeLoss(std::vector<RGBA> outputImage,
                           std::vector<RGBA> targetImage) {
  // TODO
  return 0.F;
}