#include "RunnerVisitorFactory.h"
#include "Manager.h"
#include "RunnerEnhancerVisitor.h"
#include "RunnerTrainingOpenCVVisitor.h"
#include "RunnerTrainingVulkanVisitor.h"

using namespace sipai;

const RunnerVisitor &RunnerVisitorFactory::getTrainingMonitoredVisitor() {
  if (!trainingMonitoredVisitor_) {
    const auto &app_param = Manager::getConstInstance().app_params;
    if (app_param.enable_vulkan) {
      trainingMonitoredVisitor_ =
          std::make_unique<RunnerTrainingVulkanVisitor>();
    } else {
      trainingMonitoredVisitor_ =
          std::make_unique<RunnerTrainingOpenCVVisitor>();
    }
  }
  return *trainingMonitoredVisitor_;
}

const RunnerVisitor &RunnerVisitorFactory::getEnhancerVisitor() {
  if (!enhancerVisitor_) {
    enhancerVisitor_ = std::make_unique<RunnerEnhancerVisitor>();
  }
  return *enhancerVisitor_;
}