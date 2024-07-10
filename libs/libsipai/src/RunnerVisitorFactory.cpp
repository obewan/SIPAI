#include "RunnerVisitorFactory.h"
#include "Manager.h"
#include "RunnerEnhancerOpenCVVisitor.h"
#include "RunnerEnhancerVulkanVisitor.h"
#include "RunnerTrainingOpenCVVisitor.h"
#include "RunnerTrainingVulkanVisitor.h"

using namespace sipai;

const RunnerVisitor &RunnerVisitorFactory::getTrainingVisitor() {
  if (!trainingVisitor_) {
    const auto &app_param = Manager::getConstInstance().app_params;
    if (app_param.enable_vulkan) {
      trainingVisitor_ = std::make_unique<RunnerTrainingVulkanVisitor>();
    } else {
      trainingVisitor_ = std::make_unique<RunnerTrainingOpenCVVisitor>();
    }
  }
  return *trainingVisitor_;
}

const RunnerVisitor &RunnerVisitorFactory::getEnhancerVisitor() {
  if (!enhancerVisitor_) {
    const auto &app_param = Manager::getConstInstance().app_params;
    if (app_param.enable_vulkan) {
      enhancerVisitor_ = std::make_unique<RunnerEnhancerVulkanVisitor>();
    } else {
      enhancerVisitor_ = std::make_unique<RunnerEnhancerOpenCVVisitor>();
    }
  }
  return *enhancerVisitor_;
}