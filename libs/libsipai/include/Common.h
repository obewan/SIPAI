/**
 * @file Common.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Common objects
 * @date 2024-03-14
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include "Image.h"
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <regex> // for std::regex and std::regex_replace
#include <string>
#include <unordered_set>
#include <vector>

namespace sipai {
using ImageParts = std::vector<std::unique_ptr<Image>>;
using ImagePartsPair = std::pair<ImageParts, ImageParts>;
using ImagePathPair = std::pair<std::string, std::string>;
using ImagePartsPairList = std::vector<std::pair<ImageParts, ImageParts>>;

consteval unsigned long long operator"" _K(unsigned long long x) {
  return x * 1024;
}

consteval unsigned long long operator"" _M(unsigned long long x) {
  return x * 1024_K;
}

consteval unsigned long long operator"" _G(unsigned long long x) {
  return x * 1024_M;
}

enum class ERunMode { Enhancer, Testing, Training, TrainingMonitored };

const std::map<std::string, ERunMode, std::less<>> mode_map{
    {"Enhancer", ERunMode::Enhancer},
    {"Testing", ERunMode::Testing},
    {"Training", ERunMode::Training},
    {"TrainingMonitored", ERunMode::TrainingMonitored}};

inline std::string getRunModeStr(ERunMode mode) {
  for (const auto &[key, value] : mode_map) {
    if (value == mode) {
      return key;
    }
  }
  return "";
}

/**
 * @brief Return the clamp of a cv::Mat value
 *
 * @param value
 * @param value_min
 * @param value_max
 */
inline cv::Mat mat_clamp4f(const cv::Mat &value,
                           const cv::Vec4f &value_min = cv::Vec4f::all(0.0),
                           const cv::Vec4f &value_max = cv::Vec4f::all(1.0)) {
  cv::Mat result;
  cv::min(cv::max(value, value_min), value_max, result);
  return result;
}

/**
 * @brief Return the clamp of a cv::Vec4f value
 *
 * @param value
 * @param value_min
 * @param value_max
 */
inline cv::Vec4f clamp4f(const cv::Vec4f &value,
                         const cv::Vec4f &value_min = cv::Vec4f::all(0.0),
                         const cv::Vec4f &value_max = cv::Vec4f::all(1.0)) {
  cv::Vec4f result;
  for (int i = 0; i < 4; i++) {
    result[i] = std::clamp(value[i], value_min[i], value_max[i]);
  }
  return result;
}

inline std::unordered_set<std::string> valid_extensions = {
    ".bmp",  ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png",
    ".webp", ".pbm", ".pgm",  ".ppm", ".pxm", ".pnm", ".pfm",
    ".sr",   ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic"};

/**
 * @brief Get the Filename .csv from a Filename .json
 *
 * @param filenameJson
 * @return std::string
 */
inline std::string getFilenameCsv(const std::string &filenameJson) {
  return std::regex_replace(filenameJson,
                            std::regex(".json$", std::regex::icase), ".csv");
}

inline std::array<size_t, 3> getHMSfromS(const size_t seconds) {
  size_t s = seconds;
  size_t h = s / 3600;
  s %= 3600;
  size_t m = s / 60;
  s %= 60;
  return {h, m, s};
}

} // namespace sipai