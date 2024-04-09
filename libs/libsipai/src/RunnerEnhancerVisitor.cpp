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
    const auto &app_params = manager.app_params;
    const auto &network_params = manager.network_params;
    const auto &inputImage = imageHelper_.loadImage(
        app_params.input_file, network_params.input_size_x,
        network_params.input_size_y);
    const auto &outputData = manager.network->forwardPropagation(
        inputImage.data, app_params.enable_parallel);
    Image outputImage(outputData, network_params.output_size_x,
                      network_params.output_size_y);

    imageHelper_.saveImage(app_params.output_file, outputImage,
                           outputImage.size_x * app_params.output_scale,
                           outputImage.size_y * app_params.output_scale);
    SimpleLogger::LOG_INFO("Image enhancement done. Image output saved in ",
                           manager.app_params.output_file);

  } catch (std::exception &ex) {
    throw RunnerVisitorException(ex.what());
  }
}