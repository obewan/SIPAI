/**
 * @file DataList.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief DataList
 * @date 2024-05-09
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "Data.h"
#include <vector>

namespace sipai {
enum class DataListType {
  INPUT_TARGET,
  TARGET_FOLDER,
};

struct DataList {
  std::vector<Data> data_training;
  std::vector<Data> data_validation;
  DataListType type;
};
} // namespace sipai