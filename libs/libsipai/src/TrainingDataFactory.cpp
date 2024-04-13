#include "TrainingDataFactory.h"
#include "Manager.h"
#include "exception/TrainingDataFactoryException.h"
#include <filesystem>
#include <memory>
#include <numeric>

using namespace sipai;

std::unique_ptr<TrainingDataFactory> TrainingDataFactory::instance_ = nullptr;

ImagePartsPair *TrainingDataFactory::next(
    std::vector<std::unique_ptr<ImagePathPair>> &dataPaths,
    std::vector<std::unique_ptr<ImagePartsPair>> &dataBulk,
    std::vector<std::string> &dataTargetPaths, size_t &currentIndex) {
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
  return next(dataTrainingPaths_, dataTrainingBulk_, dataTrainingTargetPaths_,
              currentTrainingIndex);
}

ImagePartsPair *TrainingDataFactory::nextValidation() {
  return next(dataValidationPaths_, dataValidationBulk_,
              dataValidationTargetPaths_, currentValidationIndex);
}

size_t TrainingDataFactory::trainingSize() { return dataTrainingPaths_.size(); }
size_t TrainingDataFactory::validationSize() {
  return dataValidationPaths_.size();
}

void TrainingDataFactory::loadData() {
  if (isLoaded_) {
    return;
  }

  const auto &app_params = Manager::getConstInstance().app_params;
  if (!app_params.training_data_file.empty()) {
    loadDataPaths();
  } else if (!app_params.training_data_folder.empty()) {
    loadDataFolder();
  } else {
    throw TrainingDataFactoryException(
        "Invalid training data file or data folder");
  }
}

void TrainingDataFactory::loadDataPaths() {
  const auto &app_params = Manager::getConstInstance().app_params;

  auto dataPaths = trainingDatafileReaderCSV_.loadTrainingDataPaths();
  splitDataPairPaths(dataPaths, app_params.training_split_ratio);

  isDataFolder = false;
  isLoaded_ = true;
}

void TrainingDataFactory::loadDataFolder() {
  const auto &app_params = Manager::getConstInstance().app_params;
  const auto &folder = app_params.training_data_folder;

  std::vector<std::string> dataTargetPaths;

  // Add images paths from the folder
  for (const auto &entry : std::filesystem::directory_iterator(folder)) {
    if (entry.is_regular_file()) {
      std::string extension = entry.path().extension().string();
      // Convert the extension to lowercase
      std::transform(extension.begin(), extension.end(), extension.begin(),
                     ::tolower);
      // Check if the file is an image by checking its extension
      if (valid_extensions.find(extension) != valid_extensions.end()) {
        dataTargetPaths.push_back(entry.path().string());
      }
    }
  }

  splitDataTargetPaths(dataTargetPaths, app_params.training_split_ratio);

  isDataFolder = true;
  isLoaded_ = true;
}

void TrainingDataFactory::resetCounters() {
  resetTraining();
  resetValidation();
}
void TrainingDataFactory::resetTraining() { currentTrainingIndex = 0; }
void TrainingDataFactory::resetValidation() { currentValidationIndex = 0; }

void TrainingDataFactory::splitDataPairPaths(
    std::vector<std::unique_ptr<ImagePathPair>> &data, float split_ratio,
    bool withRandom) {
  dataTrainingPaths_.clear();
  dataValidationPaths_.clear();

  if (data.empty()) {
    return;
  }

  if (withRandom) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
  }

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

void TrainingDataFactory::splitDataTargetPaths(std::vector<std::string> &data,
                                               float split_ratio,
                                               bool withRandom) {
  dataTrainingTargetPaths_.clear();
  dataValidationTargetPaths_.clear();

  if (data.empty()) {
    return;
  }

  if (withRandom) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data.begin(), data.end(), g);
  }

  // Calculate the split index based on the split ratio
  size_t split_index = static_cast<size_t>(data.size() * split_ratio);

  // Split the data
  for (size_t i = 0; i < data.size(); ++i) {
    if (i < split_index) {
      dataTrainingTargetPaths_.push_back(data[i]);
    } else {
      dataValidationTargetPaths_.push_back(data[i]);
    }
  }
}