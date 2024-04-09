/**
 * @file Image.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Image data struct
 * @date 2024-04-09
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "RGBA.h"
#include <cstddef>

namespace sipai {
struct Image {
  std::vector<RGBA> data;
  size_t size_x = 0;
  size_t size_y = 0;
  size_t orig_x = 0;
  size_t orig_y = 0;

  Image() {}
  Image(const std::vector<RGBA> &data, size_t size_x, size_t size_y)
      : data(data), size_x(size_x), size_y(size_y), orig_x(size_x),
        orig_y(size_y) {}
  Image(const std::vector<RGBA> &data, size_t size_x, size_t size_y,
        size_t orig_x, size_t orig_y)
      : data(data), size_x(size_x), size_y(size_y), orig_x(orig_x),
        orig_y(orig_y) {}

  size_t size() const { return data.size(); }
};
} // namespace sipai
