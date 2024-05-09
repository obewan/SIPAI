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
#include "DataList.h"
#include "ImageHelper.h"
#include "TrainingDataReader.h"
#include "exception/TrainingDataFactoryException.h"
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <random>

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
   * @brief Get the next input and target images for training.
   *
   * @return Pointer to the next input and target images
   * for training, or nullptr if no more images are available.
   */
  std::shared_ptr<Data> next(const TrainingPhase &phase);

  /**
   * @brief Get training pairs collection size
   *
   * @return size_t
   */
  size_t getSize(TrainingPhase phase) const {
    switch (phase) {
    case TrainingPhase::Training:
      return dataList_.data_training.size();
    case TrainingPhase::Validation:
      return dataList_.data_validation.size();
    default:
      throw TrainingDataFactoryException("Non-implemeted TrainingPhase");
    }
  }

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
   * @brief Indicate if the collections are loaded
   *
   * @return true
   * @return false
   */
  bool isLoaded() const { return isLoaded_; }

  /**
   * @brief Indicate if the training is using a data folder
   *
   * @return true
   * @return false
   */
  bool isDataFolder() const;

  /**
   * @brief Clear all data and reset counters
   *
   */
  void clear();

  /**
   * @brief Shuffle a vector
   *
   * @param data
   */
  void shuffle(TrainingPhase phase) {
    switch (phase) {
    case TrainingPhase::Training:
      std::shuffle(dataList_.data_training.begin(),
                   dataList_.data_training.end(), gen_);
      break;
    case TrainingPhase::Validation:
      std::shuffle(dataList_.data_validation.begin(),
                   dataList_.data_validation.end(), gen_);
    default:
      break;
    }
  }

private:
  TrainingDataFactory() : gen_(rd_()) {}
  static std::unique_ptr<TrainingDataFactory> instance_;

  TrainingDataReader trainingDataReader_;
  ImageHelper imageHelper_;
  std::atomic<bool> isLoaded_ = false;
  size_t currentTrainingIndex_ = 0;
  size_t currentValidationIndex_ = 0;

  // form random
  std::random_device rd_;
  std::mt19937 gen_;

  DataList dataList_;
};
} // namespace sipai