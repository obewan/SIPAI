/**
 * @file AppParameters.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief AppParameters
 * @date 2024-03-08
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

#include "Common.h"
#include <cstddef>
#include <string>

namespace sipai {
constexpr int NOMAX_EPOCHS = -1;

struct AppParameters {
  std::string title = "SIPAI - Simple Image Processing Artificial Intelligence";
  std::string version = "1.0.0";
  ERunMode run_mode = ERunMode::Enhancer;
  float split_ratio = 0.8;
  int max_epochs = NOMAX_EPOCHS;
  int max_epochs_without_improvement = 10;
};
} // namespace sipai