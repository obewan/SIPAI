/**
 * @file RunnerVisitorFactory.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief RunnerVisitor Factory
 * @date 2024-03-30
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "TrainingMonitoredVisitor.h"
#include <memory>

namespace sipai {
class RunnerVisitorFactory {
public:
  const RunnerVisitor &getTrainingMonitoredVisitor() {
    if (!trainingMonitoredVisitor_) {
      trainingMonitoredVisitor_ = std::make_unique<TrainingMonitoredVisitor>();
    }
    return *trainingMonitoredVisitor_;
  }
  // Add more visitors getters here

private:
  std::unique_ptr<TrainingMonitoredVisitor> trainingMonitoredVisitor_ = nullptr;
};
} // namespace sipai
