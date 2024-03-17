#include "Manager.h"
#include "Common.h"
#include "RGBA.h"
#include "SimpleLogger.h"
#include "TrainingDataFileReaderCSV.h"
#include "TrainingMonitoredVisitor.h"
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
    runWithVisitor(TrainingMonitoredVisitor{});
    break;
  default:
    break;
  }
}

void Manager::runWithVisitor(const RunnerVisitor &visitor) {

  // Load training data
  TrainingData trainingData = loadTrainingData();

  // Split training data into training and validation sets
  auto [trainingDataPairs, validationDataPairs] =
      splitData(trainingData, app_params.split_ratio);

  // Initialize network
  initializeNetwork();

  // Run the visitor
  visitor.visit(trainingDataPairs, validationDataPairs);
}

TrainingData Manager::loadTrainingData() {
  TrainingDataFileReaderCSV fileReader;
  return fileReader.getTrainingData();
}

std::pair<TrainingData, TrainingData> Manager::splitData(TrainingData data,
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