/**
 * @file RunnerTrainingOpenCVVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Concret RunnerVisitor for TrainingMonitored run.
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Common.h"
#include "ImageHelper.h"
#include "RunnerTrainingVisitor.h"
#include <memory>

namespace sipai {
class RunnerTrainingOpenCVVisitor : public RunnerTrainingVisitor {
public:
  void visit() const override;

  float trainingMonitored(size_t epoch, TrainingPhase phase) const override;

private:
  float _trainingMonitored(size_t epoch, std::shared_ptr<Data> data,
                           TrainingPhase phase, bool isLossFrequency) const;

  ImageHelper imageHelper_;
};
} // namespace sipai