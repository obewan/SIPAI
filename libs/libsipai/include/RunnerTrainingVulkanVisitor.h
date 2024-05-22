/**
 * @file RunnerTrainingVulkanVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Concret RunnerVisitor for Vulkan Training.
 * @date 2024-05-05
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "RunnerTrainingVisitor.h"

namespace sipai {
class RunnerTrainingVulkanVisitor : public RunnerTrainingVisitor {
public:
  void visit() const override;

  float computeLoss(size_t epoch, TrainingPhase phase) const;
};
} // namespace sipai