#pragma once
#include "ActivationFunctions.h"

namespace sipai {
/**
 * @brief Parameters for the neural Network.
 */
struct NetworkParameters {
  /**
   * @brief   X resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * width.
   *
   */
  size_t input_size_x = 128;
  /**
   * @brief Y resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * height.
   *
   */
  size_t input_size_y = 128;
  size_t hidden_size_x = 128;
  size_t hidden_size_y = 128;
  size_t output_size_x = 128;
  size_t output_size_y = 128;
  size_t hiddens_count = 1;
  float learning_rate = 0.01f;
  float hidden_activation_alpha = 0.1f; // used for ELU and PReLU
  float output_activation_alpha = 0.1f; // used for ELU and PReLU
  EActivationFunction hidden_activation_function = EActivationFunction::ReLU;
  EActivationFunction output_activation_function = EActivationFunction::ReLU;
};
} // namespace sipai