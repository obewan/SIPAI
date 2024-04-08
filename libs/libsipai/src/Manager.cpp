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

image Manager::loadImage(const std::string &imagePath, size_t &size_x,
                         size_t &size_y, size_t resize_x, size_t resize_y) {
  ImageHelper imageHelper;
  cv::Mat image = imageHelper.loadImage(imagePath);

  // Save the original size
  cv::Size s = image.size();
  size_x = s.width;
  size_y = s.height;

  // Resize the image to the input or output layer
  cv::resize(image, image, cv::Size(resize_x, resize_y));

  return imageHelper.convertToRGBAVector(image);
}

void Manager::saveImage(const std::string &imagePath, const image &image,
                        size_t size_x, size_t size_y, float scale) {
  saveImage(imagePath, image, size_x, size_y, size_x * scale, size_y * scale);
};

void Manager::saveImage(const std::string &imagePath, const image &image,
                        size_t size_x, size_t size_y, size_t resize_x,
                        size_t resize_y) {
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
    SimpleLogger::LOG_INFO("Saving the neural network to ",
                           app_params.network_to_export, " and ",
                           getFilenameCsv(app_params.network_to_export), "...");
    auto exportator = std::make_unique<NeuralNetworkImportExportFacade>();
    exportator->exportModel(network, network_params, app_params);
  }
}

void Manager::run() {
  SimpleLogger::LOG_INFO(getVersionHeader());

  // Initialize network
  createOrImportNetwork();

  // Run with visitor
  switch (app_params.run_mode) {
  case ERunMode::TrainingMonitored:
    runWithVisitor(runnerVisitorFactory_.getTrainingMonitoredVisitor());
    break;
  case ERunMode::Enhancer:
    runWithVisitor(runnerVisitorFactory_.getEnhancerVisitor());
    break;
  default:
    break;
  }
}

void Manager::runWithVisitor(const RunnerVisitor &visitor) { visitor.visit(); }

std::unique_ptr<TrainingData> Manager::loadTrainingData() {
  return TrainingDataFileReaderCSV{}.getTrainingData();
}

std::pair<std::unique_ptr<TrainingData>, std::unique_ptr<TrainingData>>
Manager::splitData(std::unique_ptr<TrainingData> &data, float split_ratio) {
  // Shuffle the data randomly for unbiased training and validation
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(data->begin(), data->end(), g);

  // Calculate the split index based on the split ratio
  size_t split_index = static_cast<size_t>(data->size() * split_ratio);

  // Split the data into training and validation sets
  auto training_data = std::make_unique<TrainingData>(
      data->begin(), data->begin() + split_index);
  auto validation_data =
      std::make_unique<TrainingData>(data->begin() + split_index, data->end());

  return std::make_pair(std::move(training_data), std::move(validation_data));
}
