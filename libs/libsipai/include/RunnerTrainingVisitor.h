/**
 * @file RunnerTrainingVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief RunnerTrainingVisitor
 * @date 2024-05-15
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "RunnerVisitor.h"
#include "SimpleLogger.h"
#include <csignal>
#include <mutex>

extern volatile std::sig_atomic_t stopTraining;
extern volatile std::sig_atomic_t stopTrainingNow;

extern void signalHandler(int signal);

namespace sipai {
class RunnerTrainingVisitor : public RunnerVisitor {
public:
  virtual ~RunnerTrainingVisitor() = default;

  /**
   * @brief Training and validation, including loss compute of all images.
   *
   * @param epoch the current epoch
   * @param phase indicate if it is training or validation phase
   * @return float
   */
  virtual float training(size_t epoch, TrainingPhase phase) const = 0;

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
  virtual bool shouldContinueTraining(int epoch,
                                      size_t epochsWithoutImprovement,
                                      const AppParams &appParams) const;

  /**
   * @brief Adaptive Learning Rate
   *
   * @param learningRate
   * @param validationLoss
   * @param previousValidationLoss
   * @param enable_adaptive_increase
   */
  virtual void adaptLearningRate(float &learningRate,
                                 const float &validationLoss,
                                 const float &previousValidationLoss,
                                 const bool &enable_adaptive_increase) const;

  /**
   * @brief Logs the training progress for the current epoch.
   *
   * @param epoch The current epoch number.
   * @param trainingLoss The average training loss for the current epoch.
   * @param validationLoss The average validation loss for the current epoch.
   * @param previousTrainingLoss
   * @param previousValidationLoss
   */
  virtual void logTrainingProgress(const int &epoch, const float &trainingLoss,
                                   const float &validationLoss,
                                   const float &previousTrainingLoss,
                                   const float &previousValidationLoss) const;

  /**
   * @brief Save and export the neural network
   *
   * @param hasLastEpochBeenSaved
   */
  virtual void saveNetwork(bool &hasLastEpochBeenSaved) const;

protected:
  mutable std::mutex threadMutex_;
};
} // namespace sipai