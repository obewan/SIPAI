#include "Manager.h"
#include "Common.h"
#include "ImageHelper.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkBuilder.h"
#include "SimpleLogger.h"
#include "TrainingDataFileReaderCSV.h"
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
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

void Manager::createOrImportNetwork() {
  if (!network) {
    auto builder = std::make_unique<NeuralNetworkBuilder>();
    network = builder->createOrImport()
                  .addLayers()
                  .bindLayers()
                  .addNeighbors()
                  .initializeWeights()
                  .setActivationFunction()
                  .build();
  }
}

void Manager::exportNetwork() {
  if (!app_params.network_to_export.empty()) {
    SimpleLogger::LOG_INFO("Saving the neural network, to ",
                           app_params.network_to_export, "...");
    NeuralNetworkImportExportFacade{}.exportModel();
  }
}

void Manager::run() {
  SimpleLogger::LOG_INFO(getVersionHeader());

  switch (app_params.run_mode) {
  case ERunMode::TrainingMonitored:
    runWithVisitor(runnerVisitorFactory_.getTrainingMonitoredVisitor());
    break;
  default:
    break;
  }
}

void Manager::runWithVisitor(const RunnerVisitor &visitor) {
  // Initialize network
  createOrImportNetwork();

  // Run the visitor
  visitor.visit();
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
