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
#include <functional>
#include <opencv2/core/types.hpp>
#include <sstream>
#include <string>

namespace sipai {
struct RGBA {
  std::array<float, 4> value = {0.0, 0.0, 0.0, 0.0};

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

  // Rq: OpenCV use BGRA order
  RGBA &fromVec4b(const cv::Vec4b &vec, bool hasAlpha = true) {
    value = {vec[2] / 255.0f, vec[1] / 255.0f, vec[0] / 255.0f,
             hasAlpha ? vec[3] / 255.0f : 1.0f};
    return (*this);
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
};
} // namespace sipai