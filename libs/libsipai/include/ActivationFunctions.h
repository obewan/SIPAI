/**
 * @file ActivationFunctions.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Activation functions
 * @date 2024-03-07
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include <math.h>

namespace sipai {
/**
 * @brief ReLU Function (Rectified Linear Unit): This function outputs the input
 * directly if it’s positive; otherwise, it outputs zero. It has become very
 * popular in recent years because it helps to alleviate the vanishing gradient
 * problem.
 * @param Unit
 * @return ReLU
 */
inline auto relu = [](auto x) { return std::max(0.0f, x); };
inline auto reluDerivative = [](auto x) { return x > 0 ? 1.0f : 0.0f; };

} // namespace sipai