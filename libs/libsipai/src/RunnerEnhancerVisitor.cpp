#include "RunnerEnhancerVisitor.h"
#include "ImageHelper.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "exception/RunnerVisitorException.h"
#include <exception>
#include <memory>

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

    // Load input image parts
    const auto &inputImage = imageHelper_.loadImage(
        app_params.input_file, app_params.image_split,
        app_params.enable_padding, network_params.input_size_x,
        network_params.input_size_y);

    // Get output image parts by forward propagation
    ImageParts outputParts;
    for (const auto &inputPart : inputImage) {
      const auto &outputData = manager.network->forwardPropagation(
          inputPart->data, app_params.enable_vulkan,
          app_params.enable_parallel);
      outputParts.push_back(
          std::make_unique<Image>(outputData, network_params.output_size_x,
                                  network_params.output_size_y));
    }

    // Save the output image parts as a single image
    size_t outputSizeX =
        std::accumulate(outputParts.begin(), outputParts.end(), 0,
                        [](size_t total, const std::unique_ptr<Image> &im) {
                          return total + im->size_x;
                        });
    size_t outputSizeY =
        std::accumulate(outputParts.begin(), outputParts.end(), 0,
                        [](size_t total, const std::unique_ptr<Image> &im) {
                          return total + im->size_y;
                        });
    imageHelper_.saveImage(app_params.output_file, outputParts,
                           app_params.image_split,
                           (size_t)(outputSizeX * app_params.output_scale),
                           (size_t)(outputSizeY * app_params.output_scale));

    SimpleLogger::LOG_INFO("Image enhancement done. Image output saved in ",
                           manager.app_params.output_file);

  } catch (std::exception &ex) {
    throw RunnerVisitorException(ex.what());
  }
}