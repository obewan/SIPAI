#include "TrainingDataFactory.h"
#include "Manager.h"
#include <memory>
#include <numeric>

using namespace sipai;

std::unique_ptr<TrainingDataFactory> TrainingDataFactory::instance_ = nullptr;

ImagePartsPair *TrainingDataFactory::next(
    std::vector<std::unique_ptr<ImagePathPair>> &dataPaths,
    std::vector<std::unique_ptr<ImagePartsPair>> &dataBulk,
    size_t &currentIndex) {
  const auto &manager = Manager::getConstInstance();
  const auto &app_params = manager.app_params;
  const auto &network_params = manager.network_params;

  currentIndex++;
  if (currentIndex >= dataPaths.size()) {
    // No more training data
    return nullptr;
  }

  const auto &[inputPath, targetPath] = *dataPaths[currentIndex];

  if (app_params.bulk_loading && currentIndex < dataBulk.size()) {
    // gets the data from bulk
    return dataBulk.at(currentIndex).get();
  } else {
    // load the data
    auto inputImageParts = imageHelper_.loadImage(
        inputPath, app_params.image_split, app_params.enable_padding,
        network_params.input_size_x, network_params.input_size_y);

    auto targetImageParts = imageHelper_.loadImage(
        targetPath, app_params.image_split, app_params.enable_padding,
        network_params.output_size_x, network_params.output_size_y);

    currentImagePartsPair_ = std::make_unique<ImagePartsPair>(std::make_pair(
        std::move(inputImageParts), std::move(targetImageParts)));

    // return the data, make a push in the bulk for a next time
    if (app_params.bulk_loading) {
      dataBulk.push_back(std::move(currentImagePartsPair_));
      return dataBulk.back().get();
    } else {
      return currentImagePartsPair_.get();
    }
  }
}

ImagePartsPair *TrainingDataFactory::nextTraining() {
  return next(dataTrainingPaths_, dataTrainingBulk_, currentTrainingIndex);
}

ImagePartsPair *TrainingDataFactory::nextValidation() {
  return next(dataValidationPaths_, dataValidationBulk_,
              currentValidationIndex);
}

size_t TrainingDataFactory::trainingSize() { return dataTrainingPaths_.size(); }
size_t TrainingDataFactory::validationSize() {
  return dataValidationPaths_.size();
}

void TrainingDataFactory::loadDataPaths() {
  if (isLoaded_) {
    return;
  }
  const auto &app_params = Manager::getConstInstance().app_params;

  auto dataPaths = trainingDatafileReaderCSV_.loadTrainingDataPaths();
  splitData(dataPaths, app_params.training_split_ratio);
  isLoaded_ = true;
}

void TrainingDataFactory::resetCounters() {
  resetTraining();
  resetValidation();
}
void TrainingDataFactory::resetTraining() { currentTrainingIndex = 0; }
void TrainingDataFactory::resetValidation() { currentValidationIndex = 0; }

void TrainingDataFactory::splitData(
    std::vector<std::unique_ptr<ImagePathPair>> &data, float split_ratio,
    bool withRandom) {
  if (data.empty()) {
    return;
  }

  if (withRandom) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
  }

  dataTrainingPaths_.clear();
  dataValidationPaths_.clear();

  // Calculate the split index based on the split ratio
  size_t split_index = static_cast<size_t>(data.size() * split_ratio);

  // Split the data
  for (size_t i = 0; i < data.size(); ++i) {
    if (i < split_index) {
      dataTrainingPaths_.push_back(std::move(data[i]));
    } else {
      dataValidationPaths_.push_back(std::move(data[i]));
    }
  }

  // clear the data as all its pointers has moved and are nullptr then.
  data.clear();
}