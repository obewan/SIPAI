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
#include "RunnerVisitorFactory.h"
#include <cstddef>
#include <memory>
#include <mutex>

namespace sipai {
class Manager {
public:
  static Manager &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag, [] { instance_.reset(new Manager); });
    return *instance_;
  }
  static const Manager &getConstInstance() {
    return const_cast<const Manager &>(getInstance());
  }
  Manager(Manager const &) = delete;
  void operator=(Manager const &) = delete;
  ~Manager() = default;

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
   * @brief Network builder.
   *
   * @param progressCallback
   */
  void createOrImportNetwork(std::function<void(int)> progressCallback = {});

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
   * @brief Get a title line with the version
   *
   * @return std::string
   */
  std::string getVersionHeader() const {
    return app_params.title + " v" + app_params.version;
  }

private:
  Manager() = default;

  static std::unique_ptr<Manager> instance_;

  RunnerVisitorFactory runnerVisitorFactory_;
};
} // namespace sipai