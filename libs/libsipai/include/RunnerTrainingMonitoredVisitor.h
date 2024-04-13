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
   * @brief Compute the loss of all images.
   *
   * @param isTraining indicate if it is training or validation phase
   * @return float
   */
  float computeLoss(bool isTraining) const;

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
  /**
   * @brief Compute the loss of an input image and its target image
   *
   * @param inputImage
   * @param targetImage
   * @param isTraining
   * @param isLossFrequency
   * @return float
   */
  float computeLoss(const ImageParts &inputImage, const ImageParts &targetImage,
                    bool isTraining, bool isLossFrequency) const;

  ImageHelper imageHelper_;
};
} // namespace sipai