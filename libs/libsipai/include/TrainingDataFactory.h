/**
 * @file TrainingDataFactory.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief TrainingData Factory
 * @date 2024-04-12
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "ImageHelper.h"
#include "TrainingDataFileReaderCSV.h"
#include <cstddef>
#include <memory>
#include <mutex>

namespace sipai {
class TrainingDataFactory {
public:
  static TrainingDataFactory &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { instance_.reset(new TrainingDataFactory); });
    return *instance_;
  }
  static const TrainingDataFactory &getConstInstance() {
    return const_cast<const TrainingDataFactory &>(getInstance());
  }
  TrainingDataFactory(TrainingDataFactory const &) = delete;
  void operator=(TrainingDataFactory const &) = delete;
  ~TrainingDataFactory() = default;

  ImagePartsPair *nextTraining();
  ImagePartsPair *nextValidation();

  size_t trainingSize();
  size_t validationSize();

  void loadDataPaths();
  void resetCounters();
  void resetTraining();
  void resetValidation();

  bool isLoaded() const { return isLoaded_; }

private:
  TrainingDataFactory() = default;
  static std::unique_ptr<TrainingDataFactory> instance_;

  ImagePartsPair *next(std::vector<std::unique_ptr<ImagePathPair>> &dataPaths,
                       std::vector<std::unique_ptr<ImagePartsPair>> &dataBulk,
                       size_t &currentIndex);
  void splitData(std::vector<std::unique_ptr<ImagePathPair>> &data,
                 float split_ratio, bool withRandom = false);

  std::unique_ptr<ImagePartsPair> currentImagePartsPair_ = nullptr;

  std::vector<std::unique_ptr<ImagePartsPair>> dataTrainingBulk_;
  std::vector<std::unique_ptr<ImagePartsPair>> dataValidationBulk_;

  std::vector<std::unique_ptr<ImagePathPair>> dataTrainingPaths_;
  std::vector<std::unique_ptr<ImagePathPair>> dataValidationPaths_;

  TrainingDataFileReaderCSV trainingDatafileReaderCSV_;
  ImageHelper imageHelper_;
  bool isLoaded_ = false;
  size_t currentTrainingIndex = 0;
  size_t currentValidationIndex = 0;
};
} // namespace sipai