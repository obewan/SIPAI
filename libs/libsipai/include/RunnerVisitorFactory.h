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
#include "RunnerVisitor.h"
#include <memory>

namespace sipai {
class RunnerVisitorFactory {
public:
  const RunnerVisitor &getTrainingVisitor();

  const RunnerVisitor &getEnhancerVisitor();

private:
  std::unique_ptr<RunnerVisitor> trainingVisitor_ = nullptr;
  std::unique_ptr<RunnerVisitor> enhancerVisitor_ = nullptr;
};
} // namespace sipai
