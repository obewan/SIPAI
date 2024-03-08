#pragma once
#include "ActivationFunctions.h"

namespace sipai {
/**
 * @brief Parameters for the neural Network.
 */
struct NetworkParameters {
  size_t input_size = 0;
  size_t hidden_size = 10;
  size_t output_size = 1;
  size_t hiddens_count = 1;
  float learning_rate = 0.01f;
  float hidden_activation_alpha = 0.1f; // used for ELU and PReLU
  float output_activation_alpha = 0.1f; // used for ELU and PReLU
  EActivationFunction hidden_activation_function = EActivationFunction::Sigmoid;
  EActivationFunction output_activation_function = EActivationFunction::Sigmoid;
};
} // namespace sipai