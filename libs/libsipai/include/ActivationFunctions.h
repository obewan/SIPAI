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
#include <map>
#include <math.h>
#include <string>

namespace sipai {
/**
 * @brief Activation Function enum.
 *
 */
enum class EActivationFunction { ELU, LReLU, PReLU, ReLU, Sigmoid, Tanh };

const std::map<std::string, EActivationFunction, std::less<>> activation_map{
    {"ELU", EActivationFunction::ELU},
    {"LReLU", EActivationFunction::LReLU},
    {"PReLU", EActivationFunction::PReLU},
    {"ReLU", EActivationFunction::ReLU},
    {"Sigmoid", EActivationFunction::Sigmoid},
    {"Tanh", EActivationFunction::Tanh}};

/**
 * @brief the sigmoid function is commonly used as the
 * activation function during the forward propagation step. The reason for this
 * is that the sigmoid function maps any input value into a range between 0 and
 * 1, which can be useful for outputting probabilities, among other things.
 * The sigmoid derivative can be expressed in terms of the output of
 * the sigmoid function itself: if σ(x) is the sigmoid function, then its
 * derivative σ'(x) can be computed as σ(x) * (1 - σ(x)).
 */
inline auto sigmoid = [](auto x) { return 1.0f / (1.0f + exp(-x)); };
inline auto sigmoidDerivative = [](auto x) {
  float sigmoidValue = sigmoid(x);
  return sigmoidValue * (1 - sigmoidValue);
};

/**
 * @brief Tanh Function (Hyperbolic Tangent): This function is similar to the
 * sigmoid function but maps the input to a range between -1 and 1. It is often
 * used in the hidden layers of a neural network.
 */
inline auto tanhFunc = [](auto x) { return tanh(x); };
inline auto tanhDerivative = [](auto x) {
  float tanhValue = tanh(x);
  return 1 - tanhValue * tanhValue;
};

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

/**
 * @brief Leaky ReLU: This is a variant of ReLU that allows small negative
 * values when the input is less than zero. It can help to alleviate the dying
 * ReLU problem where neurons become inactive and only output zero.
 */
inline auto leakyRelu = [](auto x) { return std::max(0.01f * x, x); };
inline auto leakyReluDerivative = [](auto x) { return x > 0 ? 1.0f : 0.01f; };

/**
 * @brief  Parametric ReLU (PReLU) is a type of leaky ReLU that, instead of
 * having a predetermined slope like 0.01, learns the slope during training.
 * This can give it a bit more flexibility and help it to learn more complex
 * patterns
 */
inline auto parametricRelu = [](auto x, auto alpha) {
  return std::max(alpha * x, x);
};
inline auto parametricReluDerivative = [](auto x, auto alpha) {
  return x > 0 ? 1.0f : alpha;
};

/**
 * @brief  the Exponential Linear Units (ELUs) are a great choice as they
 * take on negative values when the input is less than zero, which allows them
 * to push mean unit activations closer to zero like batch normalization. Unlike
 * ReLUs, ELUs have a nonzero gradient for negative input, which avoids the
 * “dead neuron” problem.
 *
 */
inline auto elu = [](auto x, auto alpha) {
  return x >= 0 ? x : alpha * (exp(x) - 1);
};
inline auto eluDerivative = [](auto x, auto alpha) {
  return x >= 0 ? 1 : alpha * exp(x);
};
} // namespace sipai