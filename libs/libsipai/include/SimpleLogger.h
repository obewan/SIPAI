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
  const SimpleLogger &log(LogLevel level, bool endl, Args &&...args) const {
    std::ostream *stream = (level == LogLevel::INFO) ? osout : oserr;
    *stream << get_timestamp() << "[" << toString(level) << "] ";
    stream->precision(current_precision);
    *stream << std::fixed;
    (*stream << ... << args);
    if (endl) {
      *stream << std::endl;
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
    osout->precision(current_precision);
    (*osout << ... << args);
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
    oserr->precision(current_precision);
    (*oserr << ... << args);
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
    osout->precision(current_precision);
    (*osout << ... << args);
    *osout << std::endl;
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
    oserr->precision(current_precision);
    (*oserr << ... << args);
    *oserr << std::endl;
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
   * @brief Outputs a newline character to the standard output
   * and flushes the stream. For INFO level only.
   * If using WARN, ERROR or DEBUG level, use endErr() instead.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  const SimpleLogger &endl() const {
    *osout << std::endl;
    osout->flush();
    return *this;
  }

  /**
   * @brief Outputs a newline character to the standard error
   * and flushes the stream. For WARN, ERROR or DEBUG level only.
   * If using INFO level, use end() instead.
   * @return const SimpleLogger& A reference to the SimpleLogger instance.
   */
  const SimpleLogger &endlErr() const {
    *oserr << std::endl;
    oserr->flush();
    return *this;
  }

  /**
   * @brief Set the Stream Out
   *
   * @param oss
   * @return const SimpleLogger&
   */
  const SimpleLogger &setStreamOut(std::ostream &oss) const {
    osout = &oss;
    return *this;
  }

  /**
   * @brief Set the Stream Err
   *
   * @param oss
   * @return const SimpleLogger&
   */
  const SimpleLogger &setStreamErr(std::ostream &oss) const {
    oserr = &oss;
    return *this;
  }

  /**
   * @brief Get the current floating precision
   *
   * @return precision
   */
  const std::streamsize &getPrecision() const { return current_precision; }

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
  SimpleLogger() {
    osout = &std::cout;
    oserr = &std::cerr;
  };
  std::streamsize default_precision = std::cout.precision();
  mutable std::streamsize current_precision = std::cout.precision();
  mutable std::mutex threadMutex_;
  mutable std::ostream *osout;
  mutable std::ostream *oserr;

  std::string get_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm{};
    std::stringstream sst;
#if defined(WIN32) || defined(_WIN32) ||                                       \
    defined(__WIN32) && !defined(__CYGWIN__)
    localtime_s(&now_tm, &now_c);
#else
    localtime_r(&now_c, &now_tm);
#endif
    sst << "[" << std::put_time(&now_tm, "%F %T") << "] ";
    return sst.str();
  }

  std::string toString(LogLevel level) const {
    switch (level) {
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARN:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
    case LogLevel::DEBUG:
      return "DEBUG";
    default:
      return "UNKNOWN";
    }
  }
};
} // namespace sipai