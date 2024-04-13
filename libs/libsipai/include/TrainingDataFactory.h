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
#include "Common.h"
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

  /**
   * @brief Get the next pair of input and target images for training.
   *
   * This method returns a pointer to the next pair of input and target images
   * for training. If there are no more images available for training, it
   * returns nullptr.
   *
   * @return ImagePartsPair* Pointer to the next pair of input and target images
   * for training, or nullptr if no more images are available.
   */
  ImagePartsPair *nextTraining();

  /**
   * @brief Get the next pair of input and target images for validation.
   *
   * This method returns a pointer to the next pair of input and target images
   * for validation. If there are no more images available for validation, it
   * returns nullptr.
   *
   * @return ImagePartsPair* Pointer to the next pair of input and target images
   * for validation, or nullptr if no more images are available.
   */
  ImagePartsPair *nextValidation();

  /**
   * @brief Get training pairs collection size
   *
   * @return size_t
   */
  size_t trainingSize();

  /**
   * @brief Get validation pairs collection size
   *
   * @return size_t
   */
  size_t validationSize();

  /**
   * @brief Load the training and validation collections paths
   *
   */
  void loadData();

  /**
   * @brief Reset training and validation counters
   *
   */
  void resetCounters();

  /**
   * @brief Reset training counter
   *
   */
  void resetTraining();

  /**
   * @brief reset validation counter
   *
   */
  void resetValidation();

  /**
   * @brief Indicate if the collections are loaded
   *
   * @return true
   * @return false
   */
  bool isLoaded() const { return isLoaded_; }

private:
  TrainingDataFactory() = default;
  static std::unique_ptr<TrainingDataFactory> instance_;

  void loadDataPaths();
  void loadDataFolder();

  ImagePartsPair *next(std::vector<std::unique_ptr<ImagePathPair>> &dataPaths,
                       std::vector<std::unique_ptr<ImagePartsPair>> &dataBulk,
                       std::vector<std::string> &dataTargetPaths,
                       size_t &currentIndex);

  void splitDataPairPaths(std::vector<std::unique_ptr<ImagePathPair>> &data,
                          float split_ratio, bool withRandom = false);

  void splitDataTargetPaths(std::vector<std::string> &data, float split_ratio,
                            bool withRandom = false);

  std::unique_ptr<ImagePartsPair> currentImagePartsPair_ = nullptr;

  // for bulk mode
  std::vector<std::unique_ptr<ImagePartsPair>> dataTrainingBulk_;
  std::vector<std::unique_ptr<ImagePartsPair>> dataValidationBulk_;

  // for csv paths file input mode
  std::vector<std::unique_ptr<ImagePathPair>> dataTrainingPaths_;
  std::vector<std::unique_ptr<ImagePathPair>> dataValidationPaths_;

  // for data folder input mode
  std::vector<std::string> dataTrainingTargetPaths_;
  std::vector<std::string> dataValidationTargetPaths_;

  TrainingDataFileReaderCSV trainingDatafileReaderCSV_;
  ImageHelper imageHelper_;
  bool isLoaded_ = false;
  bool isDataFolder = false;
  size_t currentTrainingIndex = 0;
  size_t currentValidationIndex = 0;
};
} // namespace sipai