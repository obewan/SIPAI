/**
 * @file RunnerVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief RunnerVisitor interface
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include "AppParams.h"
#include "Common.h"
#include <vector>

namespace sipai {
class RunnerVisitor {
public:
  /**
   * @brief Performs the runner operation on the provided data sets and network.
   *
   * This method is the entry point for the visitor pattern implementation. It
   * accepts the training and validation data sets, the neural network object,
   * and the application parameters, and performs the runner operation (e.g.,
   * training, inference, evaluation) accordingly.
   *
   * @param dataSet The training data set containing input-target pairs.
   * @param validationSet The validation data set containing input-target pairs.
   */
  virtual void visit(const trainingData &dataSet,
                     const trainingData &validationSet) const = 0;
};
} // namespace sipai