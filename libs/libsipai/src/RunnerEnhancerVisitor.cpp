#include "RunnerEnhancerVisitor.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"

using namespace sipai;

void RunnerEnhancerVisitor::visit() const {
  SimpleLogger::LOG_INFO("Image enhancement...");
  auto &manager = Manager::getInstance();
  if (!manager.network) {
    SimpleLogger::LOG_ERROR("No neural network. Aborting.");
    return;
  }
}