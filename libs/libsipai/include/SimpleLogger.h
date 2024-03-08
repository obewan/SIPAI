/**
 * @file SimpleLogger.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief a simple logger, using fluent interfaces.
 * @date 2023-11-02
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>

namespace sipai {
enum class LogLevel { INFO, WARN, ERROR, DEBUG };

/**
 * @brief SimpleLogger class
 *
 */
class SimpleLogger {
public:
  /**
   * @brief Get the Instance object
   * @remark This use a thread safe Meyers’ Singleton
   * @return const SimpleLogger&
   */
  const static SimpleLogger &getInstance() {
    static SimpleLogger instance;
    return instance;
  }
  SimpleLogger(SimpleLogger const &) = delete;
  void operator=(SimpleLogger const &) = delete;

  /**
   * @brief Logs messages with a timestamp and log level.
   *
   * This method logs messages with a timestamp and log level. The messages are
   * output to the standard output stream (std::cout) or to the standard error
   * stream (std::cerr).
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param level The log level of the message (INFO, WARN, ERROR, DEBUG).
   * @param endl A boolean value that determines whether a newline character is
   * appended at the end of the log message. The default value is true.
   * @param args The arguments that make up the log message. These are forwarded
   * to the output stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args>
  const SimpleLogger &log(LogLevel level, bool endl = true,
                          Args &&...args) const {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream sst;
    sst << "[" << std::put_time(std::localtime(&now_c), "%F %T") << "] ";

    switch (level) {
    case LogLevel::INFO:
      std::cout << sst.str() << "[INFO] ";
      break;
    case LogLevel::WARN:
      std::cerr << sst.str() << "[WARN] ";
      break;
    case LogLevel::ERROR:
      std::cerr << sst.str() << "[ERROR] ";
      break;
    case LogLevel::DEBUG:
      std::cerr << sst.str() << "[DEBUG] ";
      break;
    default:
      std::cerr << sst.str() << "[UNKNOWN] ";
      break;
    }

    if (level == LogLevel::INFO) {
      std::cout.precision(current_precision);
      (std::cout << ... << args);
      if (endl) {
        std::cout << std::endl;
      }
    } else {
      std::cerr.precision(current_precision);
      (std::cerr << ... << args);
      if (endl) {
        std::cerr << std::endl;
      }
    }

    return *this;
  }

  /**
   * @brief Appends messages to the standard output stream.
   *
   * This method appends messages to the standard output stream (std::cout) or
   * the standard error stream (std::cerr) without adding a newline character at
   * the end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the message. These are forwarded to
   * the output stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &append(Args &&...args) const {
    std::cout.precision(current_precision);
    (std::cout << ... << args);
    return *this;
  }

  /**
   * @brief Appends messages to the standard error stream.
   *
   * This method appends messages to the standard error stream (std::cerr)
   * without adding a newline character at the end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the message. These are forwarded to
   * the error stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args>
  const SimpleLogger &appendError(Args &&...args) const {
    std::cerr.precision(current_precision);
    (std::cerr << ... << args);
    return *this;
  }

  /**
   * @brief Outputs messages to the standard output stream with a newline
   * character at the end.
   *
   * This method outputs messages to the standard output stream (std::cout) and
   * adds a newline character at the end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the message. These are forwarded to
   * the output stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &out(Args &&...args) const {
    std::cout.precision(current_precision);
    (std::cout << ... << args);
    std::cout << std::endl;
    return *this;
  }

  /**
   * @brief Outputs messages to the standard error stream with a newline
   * character at the end.
   *
   * This method outputs messages to the standard error stream (std::cerr) and
   * adds a newline character at the end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the message. These are forwarded to
   * the error stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &err(Args &&...args) const {
    std::cerr.precision(current_precision);
    (std::cerr << ... << args);
    std::cerr << std::endl;
    return *this;
  }

  /**
   * @brief Logs messages with the INFO log level.
   *
   * This method logs messages with the INFO log level. The messages are output
   * to the standard output stream (std::cout) with a newline character at the
   * end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the log message. These are forwarded
   * to the output stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &info(Args &&...args) const {
    return log(LogLevel::INFO, true, args...);
  }

  /**
   * @brief Logs messages with the WARN log level.
   *
   * This method logs messages with the WARN log level. The messages are output
   * to the standard error stream (std::cerr) with a newline character at the
   * end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the log message. These are forwarded
   * to the error stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &warn(Args &&...args) const {
    return log(LogLevel::WARN, true, args...);
  }

  /**
   * @brief Logs messages with the ERROR log level.
   *
   * This method logs messages with the ERROR log level. The messages are output
   * to the standard error stream (std::cerr) with a newline character at the
   * end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the log message. These are forwarded
   * to the error stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &error(Args &&...args) const {
    return log(LogLevel::ERROR, true, args...);
  }

  /**
   * @brief Logs messages with the DEBUG log level.
   *
   * This method logs messages with the DEBUG log level. The messages are output
   * to the standard error stream (std::cerr) with a newline character at the
   * end.
   *
   * @tparam Args The types of the arguments that are passed to the method.
   * @param args The arguments that make up the log message. These are forwarded
   * to the error stream.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  template <typename... Args> const SimpleLogger &debug(Args &&...args) const {
    return log(LogLevel::DEBUG, true, args...);
  }

  /**
   * @brief Outputs a newline character to the standard output or error stream
   * and flushes the stream.
   *
   * This method outputs a newline character to either the standard output
   * stream (std::cout) or the standard error stream (std::cerr), based on the
   * provided argument. It then flushes the stream to ensure that the newline
   * character is immediately visible.
   *
   * @param onCerr A boolean value that determines whether the newline character
   * is output to the standard error stream. If true, the newline character is
   * output to std::cerr; otherwise, it is output to std::cout. The default
   * value is false.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  const SimpleLogger &endl(bool onCerr = false) const {
    if (onCerr) {
      std::cerr << std::endl;
      std::cerr.flush();
    } else {
      std::cout << std::endl;
      std::cout.flush();
    }
    return *this;
  }

  /**
   * @brief Set the floating precision
   *
   * @param precision
   * @return const SimpleLogger&
   */
  const SimpleLogger &setPrecision(std::streamsize precision) const {
    current_precision = precision;
    return *this;
  }

  /**
   * @brief Reset the floating precision
   *
   * @return const SimpleLogger&
   */
  const SimpleLogger &resetPrecision() const {
    current_precision = default_precision;
    return *this;
  }

  /**
   * @brief static shortcut for log info.
   * @remark thread safe
   * @tparam Args
   * @param args
   * @return const SimpleLogger&
   */
  template <typename... Args>
  static const SimpleLogger &LOG_INFO(Args &&...args) {
    auto &instance = getInstance();
    std::scoped_lock<std::mutex> lock(instance.threadMutex_);
    return instance.info(args...);
  }

  /**
   * @brief static shortcut for log warning.
   * @remark thread safe
   * @tparam Args
   * @param args
   * @return const SimpleLogger&
   */
  template <typename... Args>
  static const SimpleLogger &LOG_WARN(Args &&...args) {
    auto &instance = getInstance();
    std::scoped_lock<std::mutex> lock(instance.threadMutex_);
    return instance.warn(args...);
  }

  /**
   * @brief static shortcut for log error.
   * @remark thread safe
   * @tparam Args
   * @param args
   * @return const SimpleLogger&
   */
  template <typename... Args>
  static const SimpleLogger &LOG_ERROR(Args &&...args) {
    auto &instance = getInstance();
    std::scoped_lock<std::mutex> lock(instance.threadMutex_);
    return instance.error(args...);
  }

  /**
   * @brief static shortcut for log debug.
   * @remark thread safe
   * @tparam Args
   * @param args
   * @return const SimpleLogger&
   */
  template <typename... Args>
  static const SimpleLogger &LOG_DEBUG(Args &&...args) {
    auto &instance = getInstance();
    std::scoped_lock<std::mutex> lock(instance.threadMutex_);
    return instance.debug(args...);
  }

private:
  SimpleLogger() = default;
  std::streamsize default_precision = std::cout.precision();
  mutable std::streamsize current_precision = std::cout.precision();
  mutable std::mutex threadMutex_;
};
} // namespace sipai