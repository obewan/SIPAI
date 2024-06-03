/**
 * @file RunnerVisitor.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief RunnerVisitor interface
 * @date 2024-03-17
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include "AppParams.h"
#include "Common.h"
#include <vector>

namespace sipai {
class RunnerVisitor {
public:
  virtual ~RunnerVisitor() = default;

  /**
   * @brief Performs the runner operation on the network.
   *
   */
  virtual void visit() const = 0;
};
} // namespace sipai