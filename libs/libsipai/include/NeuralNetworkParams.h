/**
 * @file NeuralNetworkParams.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief NeuralNetworkParams
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "ActivationFunctions.h"

namespace sipai {
/**
 * @brief Parameters for the neural Network.
 */
struct NeuralNetworkParams {
  /**
   * @brief   X resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * width.
   *
   */
  size_t input_size_x = 64;
  /**
   * @brief Y resolution for input neurons. This value should not be too large
   * to avoid performance degradation. Incoming images will be resized to this
   * height.
   *
   */
  size_t input_size_y = 64;
  size_t hidden_size_x = 64;
  size_t hidden_size_y = 64;
  size_t output_size_x = 64;
  size_t output_size_y = 64;
  size_t hiddens_count = 1;
  float learning_rate = 0.005f;
  float adaptive_learning_rate_factor = 0.5f;
  float hidden_activation_alpha = 0.1f; // used for ELU and PReLU
  float output_activation_alpha = 0.1f; // used for ELU and PReLU
  bool adaptive_learning_rate = false;
  EActivationFunction hidden_activation_function = EActivationFunction::LReLU;
  EActivationFunction output_activation_function = EActivationFunction::LReLU;
};
} // namespace sipai