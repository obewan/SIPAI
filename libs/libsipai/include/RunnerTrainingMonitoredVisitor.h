/**
 * @file RunnerTrainingMonitoredVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Concret RunnerVisitor for TrainingMonitored run.
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"
#include "RGBA.h"
#include "RunnerVisitor.h"
#include <memory>

namespace sipai {
class RunnerTrainingMonitoredVisitor : public RunnerVisitor {
public:
  void visit() const override;

  /**
   * @brief Performs one epoch of training on the provided dataset.
   *
   * @param dataSet The dataset containing pairs of input and target image
   * paths.
   * @return The average loss over the training dataset for the current epoch.
   */
  float trainOnEpoch(const std::unique_ptr<TrainingData> &dataSet) const;

  /**
   * @brief Evaluates the network on the validation set.
   *
   * @param validationSet The validation set containing pairs of input and
   * target image paths.
   * @return The average loss over the validation set.
   */
  float evaluateOnValidationSet(
      const std::unique_ptr<TrainingData> &validationSet) const;

  /**
   * @brief Determines whether the training should continue based on the
   * provided conditions.
   *
   * @param epoch The current epoch number.
   * @param epochsWithoutImprovement The number of epochs without improvement in
   * validation loss.
   * @param appParams The application parameters containing the maximum number
   * of epochs and maximum epochs without improvement.
   * @return True if the training should continue, false otherwise.
   */
  bool shouldContinueTraining(int epoch, int epochsWithoutImprovement,
                              const AppParams &appParams) const;

  /**
   * @brief Logs the training progress for the current epoch.
   *
   * @param epoch The current epoch number.
   * @param trainingLoss The average training loss for the current epoch.
   * @param validationLoss The average validation loss for the current epoch.
   */
  void logTrainingProgress(int epoch, float trainingLoss,
                           float validationLoss) const;

  /**
   * @brief Save and export the neural network
   *
   * @param hasLastEpochBeenSaved
   */
  void saveNetwork(bool &hasLastEpochBeenSaved) const;

private:
  mutable std::unique_ptr<std::vector<std::pair<Image, Image>>>
      training_images_ = nullptr;
  mutable std::unique_ptr<std::vector<std::pair<Image, Image>>>
      validation_images_ = nullptr;
  std::pair<Image, Image> loadImages(const std::string &inputPath,
                                     const std::string &targetPath) const;
  std::vector<std::pair<Image, Image>>
  loadBulkImages(const std::unique_ptr<TrainingData> &dataSet,
                 std::string logPrefix) const;

  float computeLoss(const std::vector<std::pair<Image, Image>> &images,
                    bool withBackwardAndUpdateWeights) const;
  float computeLoss(const TrainingData &dataSet,
                    bool withBackwardAndUpdateWeights) const;
};
} // namespace sipai