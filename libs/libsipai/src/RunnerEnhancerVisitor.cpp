#include "RunnerEnhancerVisitor.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/RunnerVisitorException.h"
#include <exception>

using namespace sipai;

void RunnerEnhancerVisitor::visit() const {
  SimpleLogger::LOG_INFO("Image enhancement...");
  auto &manager = Manager::getInstance();

  if (!manager.network) {
    throw RunnerVisitorException("No neural network. Aborting.");
  }

  if (manager.app_params.input_file.empty()) {
    throw RunnerVisitorException("No input file. Aborting.");
  }

  if (manager.app_params.output_file.empty()) {
    throw RunnerVisitorException("No output file. Aborting.");
  }

  try {
    size_t orig_ix;
    size_t orig_iy;
    const auto &app_params = manager.app_params;
    const auto &network_params = manager.network_params;
    auto inputImage = manager.loadImage(app_params.input_file, orig_ix, orig_iy,
                                        network_params.input_size_x,
                                        network_params.input_size_y);
    auto outputImage = manager.network->forwardPropagation(
        inputImage, app_params.enable_parallel);

    manager.saveImage(app_params.output_file, outputImage, orig_ix, orig_iy,
                      app_params.output_scale);
    SimpleLogger::LOG_INFO("Image enhancement done. Image output saved in ",
                           manager.app_params.output_file);

  } catch (std::exception &ex) {
    throw RunnerVisitorException(ex.what());
  }
}