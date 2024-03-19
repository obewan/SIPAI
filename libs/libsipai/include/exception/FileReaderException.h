
#pragma once
#include <exception>
#include <string>

namespace sipai {
/**
 * @brief A custom exception class that inherits from std::exception.
 * This class is thrown when there are issues with file reader operations.
 */
class FileReaderException : public std::exception {
public:
  explicit FileReaderException(const std::string &message)
      : message_(message) {}
  const char *what() const noexcept override { return message_.c_str(); }

private:
  std::string message_;
};
} // namespace sipai