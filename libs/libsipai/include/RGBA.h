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

namespace sipai {
struct RGBA {
  std::array<float, 4> value;

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