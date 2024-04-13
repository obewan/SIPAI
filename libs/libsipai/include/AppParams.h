/**
 * @file AppParams.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief AppParams
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
constexpr int NO_MAX_EPOCHS = -1;

struct AppParams {
  std::string title = "SIPAI - Simple Image Processing Artificial Intelligence";
  std::string version = "0.0.1";
  std::string input_file = "";
  std::string output_file = "";
  std::string training_data_file = "";
  std::string training_data_folder = "";
  std::string network_to_import = "";
  std::string network_to_export = "";
  ERunMode run_mode = ERunMode::Enhancer;
  float output_scale = 1.0;
  float training_split_ratio = 0.7;
  int max_epochs = NO_MAX_EPOCHS;
  size_t max_epochs_without_improvement = 2;
  size_t epoch_autosave = 100;
  size_t image_split = 1;
  size_t training_reduce_factor = 4;
  bool bulk_loading = false;
  bool enable_parallel = false;
  bool enable_padding = false;
};
} // namespace sipai