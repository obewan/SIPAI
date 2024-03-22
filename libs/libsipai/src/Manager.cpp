#include "Manager.h"
#include "Common.h"
#include "ImageHelper.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkImportExportFacade.h"
#include "SimpleLogger.h"
#include "TrainingDataFileReaderCSV.h"
#include "TrainingMonitoredVisitor.h"
#include <cstddef>
#include <memory>
#include <opencv2/core/types.hpp>
#include <vector>

using namespace sipai;

std::vector<RGBA> Manager::loadImage(const std::string &imagePath,
                                     size_t &size_x, size_t &size_y,
                                     size_t resize_x, size_t resize_y) {
  ImageHelper imageHelper;
  cv::Mat image = imageHelper.loadImage(imagePath);

  // Save the original size
  cv::Size s = image.size();
  size_x = s.width;
  size_y = s.height;

  // Resize the image to the neurons layer
  cv::resize(image, image, cv::Size(resize_x, resize_y));

  return imageHelper.convertToRGBAVector(image);
}

void Manager::saveImage(const std::string &imagePath,
                        const std::vector<RGBA> &image, size_t size_x,
                        size_t size_y) {
  saveImage(imagePath, image, size_x, size_y, size_x, size_y);
};

void Manager::saveImage(const std::string &imagePath,
                        const std::vector<RGBA> &image, size_t size_x,
                        size_t size_y, size_t resize_x, size_t resize_y) {
  ImageHelper imageHelper;
  cv::Mat dest = imageHelper.convertToMat(image, size_x, size_y);
  cv::resize(dest, dest, cv::Size(resize_x, resize_y));
  imageHelper.saveImage(imagePath, dest);
}

void Manager::createNetwork() { network = std::make_unique<NeuralNetwork>(); }

void Manager::importNetwork() {
  network = NeuralNetworkImportExportFacade{}.importModel();
}

void Manager::exportNetwork() {
  NeuralNetworkImportExportFacade{}.exportModel();
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
  return TrainingDataFileReaderCSV{}.getTrainingData();
}

std::pair<TrainingData, TrainingData> Manager::splitData(TrainingData data,
                                                         float split_ratio) {
  // Shuffle the data randomly for unbiased training and validation
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data.begin(), data.end(), g);

  // Calculate the split index based on the split ratio
  size_t split_index = static_cast<size_t>(data.size() * split_ratio);

  // Split the data into training and validation sets
  TrainingData training_data(data.begin(), data.begin() + split_index);
  TrainingData validation_data(data.begin() + split_index, data.end());

  return std::make_pair(training_data, validation_data);
}

void Manager::initializeNetwork() { network->initialize(); }

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