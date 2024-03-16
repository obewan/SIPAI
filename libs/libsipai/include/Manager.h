/**
 * @file Manager.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Manager
 * @date 2024-03-08
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "AppParameters.h"
#include "Common.h"
#include "Network.h"
#include "NetworkParameters.h"
#include "RGBA.h"
#include <memory>

namespace sipai {
class Manager {
public:
  static Manager &getInstance() {
    static Manager instance;
    return instance;
  }
  Manager(Manager const &) = delete;
  void operator=(Manager const &) = delete;

  /**
   * @brief Application parameters.
   */
  AppParameters app_params;

  /**
   * @brief Network parameters.
   */
  NetworkParameters network_params;

  /**
   * @brief The neural network
   */
  std::shared_ptr<Network> network = nullptr;

  /**
   * @brief Load an image and convert it for the input layer
   *
   * @param imagePath
   */
  std::vector<RGBA> loadImage(const std::string &imagePath);

  /**
   * @brief Run the ai
   *
   */
  void run();

  /**
   * @brief Runs the training process in monitored mode.
   *
   * This method loads the training data, splits it into training and validation
   * sets, initializes the neural network, and then enters a training loop.
   * During each epoch, it performs forward propagation, computes the loss,
   * performs backward propagation, and updates the network weights using the
   * training data. It also evaluates the model on the validation set and logs
   * the training progress. Early stopping is implemented to prevent
   * overfitting.
   */
  void runTrainingMonitored();

  /**
   * @brief Loads the training data from the specified source.
   *
   * @return A vector of pairs, where each pair contains the paths to the input
   * image and the corresponding target image.
   */
  trainingData loadTrainingData();

  /**
   * @brief Performs one epoch of training on the provided dataset.
   *
   * @param dataSet The dataset containing pairs of input and target image
   * paths.
   * @return The average loss over the training dataset for the current epoch.
   */
  float trainOnEpoch(const trainingData &dataSet);

  /**
   * @brief Evaluates the network on the validation set.
   *
   * @param validationSet The validation set containing pairs of input and
   * target image paths.
   * @return The average loss over the validation set.
   */
  float evaluateOnValidationSet(const trainingData &validationSet);

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
                              const AppParameters &appParams);

  /**
   * @brief Logs the training progress for the current epoch.
   *
   * @param epoch The current epoch number.
   * @param trainingLoss The average training loss for the current epoch.
   * @param validationLoss The average validation loss for the current epoch.
   */
  void logTrainingProgress(int epoch, float trainingLoss, float validationLoss);

  /**
   * @brief Splits the training data into training and validation sets.
   *
   * @param data The training data to be split.
   * @param split_ratio The ratio of the data to be used for the training
   * set. For example, if split_ratio is 0.8, 80% of the data will be used
   * for the training set, and the remaining 20% will be used for the
   * validation set.
   *
   * @return A pair of vectors, where the first element is the training
   * data, and the second element is the validation data.
   */
  std::pair<trainingData, trainingData> splitData(trainingData data,
                                                  float split_ratio);

  /**
   * @brief Initializes the neural network architecture.
   *
   * This method sets up the input, hidden, and output layers of the neural
   * network, along with any necessary configurations or parameters.
   */
  void initializeNetwork();

  /**
   * @brief Computes the mean squared error (MSE) loss between the output image
   * and the target image.
   *
   * @param outputImage The output image produced by the neural network.
   * @param targetImage The expected target image.
   *
   * @return The computed MSE loss.
   */
  float computeMSELoss(const std::vector<RGBA> &outputImage,
                       const std::vector<RGBA> &targetImage);

  /**
   * @brief Get a title line with the version
   *
   * @return std::string
   */
  std::string getVersionHeader() const {
    return app_params.title + " v" + app_params.version;
  }

private:
  Manager() = default;
};
} // namespace sipai