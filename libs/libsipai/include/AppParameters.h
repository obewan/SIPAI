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

#include <cstddef>
#include <string>

namespace sipai {
struct AppParameters {
  std::string title = "SIPAI - Simple Image Processing Artificial Intelligence";
  std::string version = "1.0.0";
  /**
   * @brief   X resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * width.
   *
   */
  size_t input_res_x = 64;
  /**
   * @brief Y resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * height.
   *
   */
  size_t input_res_y = 64;
};
} // namespace sipai