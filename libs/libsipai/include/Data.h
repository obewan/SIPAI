/**
 * @file Data.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Data
 * @date 2024-05-09
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Image.h"
#include <memory>
#include <string>

namespace sipai {
using ImageParts = std::vector<std::shared_ptr<Image>>;

struct Data {
  std::string file_input;
  std::string file_target;
  std::string file_output;
  ImageParts img_input;
  ImageParts img_target;
  ImageParts img_output;
};
} // namespace sipai