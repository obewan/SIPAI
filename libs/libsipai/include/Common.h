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

} // namespace sipai