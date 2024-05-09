#include "TrainingDataFactory.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/TrainingDataFactoryException.h"
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>

using namespace sipai;

std::unique_ptr<TrainingDataFactory> TrainingDataFactory::instance_ = nullptr;

bool TrainingDataFactory::isDataFolder() const {
  const auto &app_params = Manager::getConstInstance().app_params;
  return !app_params.training_data_folder.empty() &&
         app_params.training_data_file.empty();
}

void TrainingDataFactory::loadData() {
  if (isLoaded_) {
    return;
  }

  const auto &app_params = Manager::getConstInstance().app_params;
  if (app_params.verbose) {
    SimpleLogger::LOG_INFO("Loading images paths...");
  }

  std::vector<Data> datas;
  // load images paths
  if (!app_params.training_data_file.empty()) {
    datas = trainingDataReader_.loadTrainingDataPaths();
    dataList_.type = DataListType::INPUT_TARGET;
  } else if (!app_params.training_data_folder.empty()) {
    datas = trainingDataReader_.loadTrainingDataFolder();
    dataList_.type = DataListType::TARGET_FOLDER;
  } else {
    throw TrainingDataFactoryException(
        "Invalid training data file or data folder");
  }
  if (app_params.random_loading) {
    std::shuffle(datas.begin(), datas.end(), gen_);
  }
  // split datas
  size_t split_index =
      static_cast<size_t>(datas.size() * app_params.training_split_ratio);
  for (size_t i = 0; i < datas.size(); ++i) {
    if (i < split_index) {
      dataList_.data_training.push_back(datas[i]);
    } else {
      dataList_.data_validation.push_back(datas[i]);
    }
  }

  isLoaded_ = true;
  if (app_params.verbose) {
    SimpleLogger::LOG_INFO(
        "Images paths loaded: ", dataList_.data_training.size(),
        " images for training, ", dataList_.data_validation.size(),
        " images for validation.");
  }
}

std::shared_ptr<Data> TrainingDataFactory::next(const TrainingPhase &phase) {
  const auto &manager = Manager::getConstInstance();
  const auto &app_params = manager.app_params;
  const auto &network_params = manager.network_params;
  size_t *index = nullptr;
  std::vector<Data> *datas = nullptr;
  switch (phase) {
  case TrainingPhase::Training:
    index = &currentTrainingIndex_;
    datas = &dataList_.data_training;
    break;
  case TrainingPhase::Validation:
    index = &currentValidationIndex_;
    datas = &dataList_.data_validation;
    break;
  default:
    throw TrainingDataFactoryException("Unimplemented TrainingPhase");
  }

  if (*index >= datas->size()) {
    // No more training data
    return nullptr;
  }
  auto &data = datas->at(*index);
  // check if bulk_loading and already loaded
  if (app_params.bulk_loading && data.img_input.size() > 0 &&
      data.img_output.size() > 0) {
    return std::make_shared<Data>(data);
  }

  // load the target image
  ImageParts targetImageParts = imageHelper_.loadImage(
      data.file_target, app_params.image_split, app_params.enable_padding,
      network_params.output_size_x, network_params.output_size_y);

  // generate or load the input image
  ImageParts inputImageParts =
      dataList_.type == DataListType::TARGET_FOLDER
          ? imageHelper_.generateInputImage(
                targetImageParts, app_params.training_reduce_factor,
                network_params.input_size_x, network_params.input_size_y)
          : imageHelper_.loadImage(data.file_input, app_params.image_split,
                                   app_params.enable_padding,
                                   network_params.input_size_x,
                                   network_params.input_size_y);

  (*index)++;

  if (app_params.bulk_loading) {
    data.img_input = inputImageParts;
    data.img_target = targetImageParts;
    return std::make_shared<Data>(data);
  } else {
    return std::make_shared<Data>(Data{
        .file_input = data.file_input,
        .file_output = data.file_output,
        .file_target = data.file_target,
        .img_input = inputImageParts,
        .img_output = data.img_output,
        .img_target = targetImageParts,
    });
  }
}

void TrainingDataFactory::resetCounters() {
  currentTrainingIndex_ = 0;
  currentValidationIndex_ = 0;
}

void TrainingDataFactory::clear() {
  dataList_.data_training.clear();
  dataList_.data_validation.clear();
  resetCounters();
  isLoaded_ = false;
}