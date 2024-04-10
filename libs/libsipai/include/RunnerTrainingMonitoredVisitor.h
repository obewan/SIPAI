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
#include "Image.h"
#include "ImageHelper.h"
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
  bool shouldContinueTraining(int epoch, size_t epochsWithoutImprovement,
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
  ImageHelper imageHelper_;
  mutable std::unique_ptr<std::vector<std::pair<ImageParts, ImageParts>>>
      training_images_ = nullptr;
  mutable std::unique_ptr<std::vector<std::pair<ImageParts, ImageParts>>>
      validation_images_ = nullptr;

  std::pair<ImageParts, ImageParts>
  loadImages(const std::string &inputPath, const std::string &targetPath) const;
  std::vector<std::pair<ImageParts, ImageParts>>
  loadBulkImages(const std::unique_ptr<TrainingData> &dataSet,
                 std::string logPrefix) const;

  /**
   * @brief Compute the loss of an input image and its target image
   *
   * @param inputImage
   * @param targetImage
   * @param withBackwardAndUpdateWeights
   * @param isLossFrequency
   * @return float
   */
  float computeLoss(const ImageParts &inputImage, const ImageParts &targetImage,
                    bool withBackwardAndUpdateWeights,
                    bool isLossFrequency) const;
  /**
   * @brief Compute the loss of loaded images, without unloading.
   *
   * @param images
   * @param withBackwardAndUpdateWeights
   * @return float
   */
  float
  computeLoss(const std::vector<std::pair<ImageParts, ImageParts>> &images,
              bool withBackwardAndUpdateWeights) const;
  /**
   * @brief Compute the loss of a data set of image paths, loading/unloading
   * image one by one for low memory usage
   *
   * @param dataSet
   * @param withBackwardAndUpdateWeights
   * @return float
   */
  float computeLoss(const TrainingData &dataSet,
                    bool withBackwardAndUpdateWeights) const;
};
} // namespace sipai