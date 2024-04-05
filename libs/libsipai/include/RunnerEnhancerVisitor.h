/**
 * @file RunnerEnhancerVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Concret RunnerVisitor for Enhancer run.
 * @date 2024-04-05
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"
#include "RunnerVisitor.h"

namespace sipai {
class RunnerEnhancerVisitor : public RunnerVisitor {
public:
  void visit() const override;
};
} // namespace sipai