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
#include "RunnerEnhancerVisitor.h"
#include "RunnerTrainingMonitoredVisitor.h"
#include <memory>

namespace sipai {
class RunnerVisitorFactory {
public:
  const RunnerVisitor &getTrainingMonitoredVisitor() {
    if (!trainingMonitoredVisitor_) {
      trainingMonitoredVisitor_ =
          std::make_unique<RunnerTrainingMonitoredVisitor>();
    }
    return *trainingMonitoredVisitor_;
  }

  const RunnerVisitor &getEnhancerVisitor() {
    if (!enhancerVisitor_) {
      enhancerVisitor_ = std::make_unique<RunnerEnhancerVisitor>();
    }
    return *enhancerVisitor_;
  }

private:
  std::unique_ptr<RunnerTrainingMonitoredVisitor> trainingMonitoredVisitor_ =
      nullptr;
  std::unique_ptr<RunnerEnhancerVisitor> enhancerVisitor_ = nullptr;
};
} // namespace sipai
