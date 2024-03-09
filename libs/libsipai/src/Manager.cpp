#include "Manager.h"

using namespace sipai;

std::vector<RGBA> Manager::loadImage(const std::string &imagePath) {
  ImageImport imageImport;
  cv::Mat image = imageImport.importImage(imagePath);

  // Resize the image to the input neurons
  const auto &app_params = Manager::getInstance().app_params;
  cv::resize(image, image,
             cv::Size(app_params.input_res_x, app_params.input_res_y));

  return imageImport.convertToRGBAVector(image);
}