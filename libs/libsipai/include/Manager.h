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

#include "AppParams.h"
#include "Common.h"
#include "NeuralNetworkBuilder.h"
#include "NeuralNetworkParams.h"
#include "RGBA.h"
#include "RunnerVisitorFactory.h"
#include <cstddef>
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
  AppParams app_params;

  /**
   * @brief Network parameters.
   */
  NeuralNetworkParams network_params;

  /**
   * @brief The neural network.
   */
  std::unique_ptr<NeuralNetwork> network = nullptr;

  /**
   * @brief Loads an image from a specified path and resizes it for the input
   * layer.
   *
   * @param imagePath The file path of the image to be loaded.
   * @param size_x A reference to a variable where the original image width will
   * be stored.
   * @param size_y A reference to a variable where the original image height
   * will be stored.
   * @param resize_x The desired width to which the image should be resized.
   * @param resize_y The desired height to which the image should be resized.
   * @return A vector of RGBA values representing the resized image.
   */
  std::vector<RGBA> loadImage(const std::string &imagePath, size_t &size_x,
                              size_t &size_y, size_t resize_x, size_t resize_y);

  /**
   * @brief Exports a resized image to a specified path.
   *
   * @param imagePath The file path where the image will be exported.
   * @param image A vector of RGBA values representing the image to be exported.
   * @param size_x The original width of the image represented by the vector of
   * RGBA values.
   * @param size_y The original height of the image represented by the vector of
   * RGBA values.
   */
  void saveImage(const std::string &imagePath, const std::vector<RGBA> &image,
                 size_t size_x, size_t size_y);
  /**
   * @brief Exports a resized image to a specified path.
   *
   * @param imagePath The file path where the image will be exported.
   * @param image A vector of RGBA values representing the image to be exported.
   * @param size_x The original width of the image represented by the vector of
   * RGBA values.
   * @param size_y The original height of the image represented by the vector of
   * RGBA values.
   * @param resize_x The desired width to which the image should be resized
   * before exporting.
   * @param resize_y The desired height to which the image should be resized
   * before exporting.
   */
  void saveImage(const std::string &imagePath, const std::vector<RGBA> &image,
                 size_t size_x, size_t size_y, size_t resize_x,
                 size_t resize_y);

  /**
   * @brief Network builder.
   */
  void createOrImportNetwork();

  /**
   * @brief Export the neural network to its json and csv files.
   *
   */
  void exportNetwork();

  /**
   * @brief Run the ai (main entrance).
   *
   */
  void run();

  /**
   * @brief Runs the provided visitor on the training and validation data sets
   * with the initialized neural network.
   *
   * This method is the entry point for executing different types of runners
   * (e.g., training, inference, evaluation) on the loaded data sets and the
   * initialized neural network. It accepts a visitor object implementing the
   * `RunnerVisitor` interface and calls its `visit` method, passing the
   * necessary data and parameters.
   *
   * @param visitor The visitor object implementing the `RunnerVisitor`
   * interface, which encapsulates the runner logic.
   */
  void runWithVisitor(const RunnerVisitor &visitor);

  /**
   * @brief Loads the training data from the specified source.
   *
   * @return A vector of pairs, where each pair contains the paths to the input
   * image and the corresponding target image.
   */
  TrainingData loadTrainingData();

  /**
   * @brief Shuffle and splits the training data into training and validation
   * sets.
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
  std::pair<TrainingData, TrainingData> splitData(TrainingData data,
                                                  float split_ratio);

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

  RunnerVisitorFactory runnerVisitorFactory_;
  NeuralNetworkBuilder neuralNetworkBuilder_;
};
} // namespace sipai