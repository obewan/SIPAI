/**
 * @file RunnerEnhancerVulkanVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Concret RunnerVisitor for Enhancer run.
 * @date 2024-07-10
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"
#include "ImageHelper.h"
#include "RunnerVisitor.h"

namespace sipai {
class RunnerEnhancerVulkanVisitor : public RunnerVisitor {
public:
  void visit() const override;

private:
  ImageHelper imageHelper_;
};
} // namespace sipai