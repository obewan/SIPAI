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
#include "RGBA.h"
#include <map>
#include <regex> // for std::regex and std::regex_replace
#include <string>
#include <vector>

namespace sipai {
using TrainingData = std::vector<std::pair<std::string, std::string>>;
using image = std::vector<RGBA>;

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