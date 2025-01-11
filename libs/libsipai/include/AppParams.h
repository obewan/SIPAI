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
constexpr int NO_MAX_EPOCHS = 0;
constexpr int NO_IMAGE_SPLIT = 0;


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
  float output_scale = 1.0f;
  float training_split_ratio = 0.7f;
  float learning_rate_max = 1.0f;
  float learning_rate_min = 0.00001f;
  size_t max_epochs = NO_MAX_EPOCHS;
  size_t max_epochs_without_improvement = 2; // TODO: check for 0 = no max
  size_t epoch_autosave = 100;               // TODO: check for 0 = no autosave
  size_t image_split = NO_IMAGE_SPLIT;
  size_t training_reduce_factor = 4;
  bool random_loading = false;
  bool bulk_loading = false;
  bool enable_vulkan = false;
  bool enable_parallel = true;
  bool enable_padding = false;
  bool no_save = false;
  bool verbose = false;
  bool verbose_debug = false;
  bool vulkan_debug = false;
};
} // namespace sipai