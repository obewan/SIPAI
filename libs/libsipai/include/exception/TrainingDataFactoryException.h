
#pragma once
#include "TrainingDataFactory.h"
#include <exception>
#include <string>

namespace sipai {
/**
 * @brief A custom exception class that inherits from std::exception.
 * This class is thrown when there are issues with TrainingDataFactory
 * operations.
 */
class TrainingDataFactoryException : public std::exception {
public:
  explicit TrainingDataFactoryException(const std::string &message)
      : message_(message) {}
  const char *what() const noexcept override { return message_.c_str(); }

private:
  std::string message_;
};
} // namespace sipai