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
#include "Common.h"
#include <algorithm>
#include <map>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace sipai {
/**
 * @brief Activation Function enum.
 * Beware the int values are used in the Vulkan GLSL shader
 */
enum class EActivationFunction {
  ELU = 0,
  LReLU = 1,
  PReLU = 2,
  ReLU = 3,
  Sigmoid = 4,
  Tanh = 5
};

const std::map<std::string, EActivationFunction, std::less<>> activation_map{
    {"ELU", EActivationFunction::ELU},
    {"LReLU", EActivationFunction::LReLU},
    {"PReLU", EActivationFunction::PReLU},
    {"ReLU", EActivationFunction::ReLU},
    {"Sigmoid", EActivationFunction::Sigmoid},
    {"Tanh", EActivationFunction::Tanh}};

inline std::string getActivationStr(EActivationFunction activation) {
  for (const auto &[key, value] : activation_map) {
    if (value == activation) {
      return key;
    }
  }
  return "";
}

/**
 * @brief the sigmoid function is commonly used as the
 * activation function during the forward propagation step. The reason for this
 * is that the sigmoid function maps any input value into a range between 0 and
 * 1, which can be useful for outputting probabilities, among other things.
 * The sigmoid derivative can be expressed in terms of the output of
 * the sigmoid function itself: if σ(x) is the sigmoid function, then its
 * derivative σ'(x) can be computed as σ(x) * (1 - σ(x)).
 */
inline auto sigmoid = [](const cv::Vec4f &rgba) {
  cv::Vec4f result;
  cv::exp(-rgba, result);
  cv::divide(cv::Vec4f::all(1.0f), (cv::Vec4f::all(1.0f) + result), result);
  return Common::clamp4f(result);
};

inline auto sigmoidDerivative = [](const cv::Vec4f &rgba) {
  cv::Vec4f sigmoidValue = sigmoid(rgba);
  return sigmoidValue.mul(cv::Vec4f::all(1.0f) - sigmoidValue);
};

/**
 * @brief Tanh Function (Hyperbolic Tangent): This function is similar to the
 * sigmoid function but maps the input to a range between -1 and 1. It is often
 * used in the hidden layers of a neural network.
 */
inline auto tanhFunc = [](const cv::Vec4f &rgba) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val,
                 [](float v) { return (std::tanh(v) / 2.0f) + 0.5f; });
  return result;
};

inline auto tanhDerivative = [](const cv::Vec4f &rgba) {
  cv::Vec4f tanhValue = tanhFunc(rgba);
  return cv::Vec4f::all(1.0f) - tanhValue.mul(tanhValue);
};

/**
 * @brief ReLU Function (Rectified Linear Unit): This function outputs the input
 * directly if it’s positive; otherwise, it outputs zero. It has become very
 * popular in recent years because it helps to alleviate the vanishing gradient
 * problem.
 * Combine ReLU with clamping to [0, 1] range
 * @param rgba
 * @return ReLU
 */

inline auto relu = [](const cv::Vec4f &rgba) { return Common::clamp4f(rgba); };
inline auto reluDerivative = [](const cv::Vec4f &rgba) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val,
                 [](float v) { return v > 0.0f ? 1.0f : 0.0f; });
  return result;
};

/**
 * @brief Leaky ReLU: This is a variant of ReLU that allows small negative
 * values when the input is less than zero. It can help to alleviate the dying
 * ReLU problem where neurons become inactive and only output zero.
 * Combine LReLU with clamping to [0, 1] range
 */
inline auto leakyRelu = [](const cv::Vec4f &rgba) {
  return Common::clamp4f(rgba * 0.01f);
};

inline auto leakyReluDerivative = [](const cv::Vec4f &rgba) {
  return Common::clamp4f(rgba, cv::Vec4f::all(0.01f), cv::Vec4f::all(1.0f));
};

/**
 * @brief  Parametric ReLU (PReLU) is a type of leaky ReLU that, instead of
 * having a predetermined slope like 0.01, learns the slope during training.
 * This can give it a bit more flexibility and help it to learn more complex
 * patterns
 */
inline auto parametricRelu = [](const cv::Vec4f &rgba, float alpha) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val, [alpha](float v) {
    float value = std::max(alpha * v, v);
    return std::clamp(value, 0.f, 1.f);
  });
  return result;
};

inline auto parametricReluDerivative = [](const cv::Vec4f &rgba, float alpha) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val,
                 [alpha](float v) { return v > 0.0f ? 1.0f : alpha; });
  return result;
};

/**
 * @brief  the Exponential Linear Units (ELUs) are a great choice as they
 * take on negative values when the input is less than zero, which allows them
 * to push mean unit activations closer to zero like batch normalization. Unlike
 * ReLUs, ELUs have a nonzero gradient for negative input, which avoids the
 * “dead neuron” problem.
 *
 */
inline auto elu = [](const cv::Vec4f &rgba, float alpha) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val, [alpha](float v) {
    float value = v >= 0 ? v : alpha * (exp(v) - 1);
    return std::clamp(value, 0.f, 1.f);
  });
  return result;
};

inline auto eluDerivative = [](const cv::Vec4f &rgba, float alpha) {
  cv::Vec4f result;
  std::transform(rgba.val, rgba.val + 4, result.val,
                 [alpha](float v) { return v > 0.0f ? 1.0f : alpha * exp(v); });
  return result;
};
} // namespace sipai