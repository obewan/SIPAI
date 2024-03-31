/**
 * @file RGBA.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief
 * @date 2024-03-10
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <math.h>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <string>

namespace sipai {
struct RGBA {
  std::array<float, 4> value = {0.0, 0.0, 0.0, 0.0};

  // Default constructor
  RGBA() : value({0.0f, 0.0f, 0.0f, 0.0f}) {}

  // Parameterized constructor
  RGBA(float r, float g, float b, float a) : value({r, g, b, a}) {}

  // Rq: OpenCV use BGRA order
  RGBA(const cv::Vec4b &vec, bool hasAlpha = true)
      : value({vec[2] / 255.0f, vec[1] / 255.0f, vec[0] / 255.0f,
               hasAlpha ? vec[3] / 255.0f : 1.0f}) {}

  float sum() const {
    return std::reduce(value.begin(), value.end(), 0.0f, std::plus<>());
  }

  void clip(float lower, float upper) {
    for (auto &val : value) {
      val = std::max(lower, std::min(val, upper));
    }
  }

  RGBA pow(float n) {
    if (n < 1.0 && std::any_of(value.begin(), value.end(),
                               [](float v) { return v < 0.0f; })) {
      throw std::invalid_argument(
          "Negative value in RGBA for non-integer exponent");
    }
    return RGBA(std::pow(value[0], n), std::pow(value[1], n),
                std::pow(value[2], n), std::pow(value[3], n));
  }

  std::string toStringCsv() const {
    std::ostringstream oss;
    oss << std::fixed; // avoid scientific notation
    for (float v : value) {
      oss << v << ",";
    }
    oss.seekp(-1, std::ios_base::end); // Remove the trailing comma
    return oss.str();
  }

  // Rq: OpenCV use BGRA order
  cv::Vec4b toVec4b() const {
    return cv::Vec4b(value[2] * 255, value[1] * 255, value[0] * 255,
                     value[3] * 255);
  }

  // Helper method to apply a lambda function to each component
  RGBA apply(const std::function<float(float)> &func) const {
    RGBA result;
    std::transform(value.begin(), value.end(), result.value.begin(), func);
    return result;
  }

  // Overloaded helper method to apply a lambda function with an additional
  // parameter
  RGBA apply(const std::function<float(float, float)> &func,
             float param) const {
    RGBA result;
    std::transform(value.begin(), value.end(), result.value.begin(),
                   [&func, param](float v) { return func(v, param); });
    return result;
  }

  // Define a helper function to apply a lambda to each element
  auto applyToElements(auto func, const RGBA &rhs) const {
    RGBA result;
    std::transform(this->value.begin(), this->value.end(), rhs.value.begin(),
                   result.value.begin(), func);
    return result;
  }

  RGBA &operator+=(const RGBA &rhs) {
    std::transform(this->value.begin(), this->value.end(), rhs.value.begin(),
                   this->value.begin(), std::plus<>());
    return *this;
  }

  RGBA &operator-=(const RGBA &rhs) {
    std::transform(this->value.begin(), this->value.end(), rhs.value.begin(),
                   this->value.begin(), std::minus<>());
    return *this;
  }

  RGBA &operator*=(const RGBA &rhs) {
    std::transform(this->value.begin(), this->value.end(), rhs.value.begin(),
                   this->value.begin(), std::multiplies<>());
    return *this;
  }

  RGBA operator*(const RGBA &rhs) const {
    return applyToElements(std::multiplies<>(), rhs);
  }

  RGBA operator+(const RGBA &rhs) const {
    return applyToElements(std::plus<>(), rhs);
  }

  RGBA operator-(const RGBA &rhs) const {
    return applyToElements(std::minus<>(), rhs);
  }

  friend RGBA operator*(float lhs, const RGBA &rhs) {
    RGBA result;
    std::transform(rhs.value.begin(), rhs.value.end(), result.value.begin(),
                   [&lhs](float val) { return lhs * val; });
    return result;
  }
}; // namespace sipai
} // namespace sipai