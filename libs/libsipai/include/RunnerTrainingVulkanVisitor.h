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
#include "RunnerVisitor.h"

namespace sipai {
class RunnerTrainingVulkanVisitor : public RunnerVisitor {
public:
  void visit() const override;
};
} // namespace sipai