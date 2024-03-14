/**
 * @file Sipai.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Sipai
 * @date 2024-03-08
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once

class SIPAI {
public:
  static const int EXIT_HELP = 2;
  static const int EXIT_VERSION = 3;

  /**
   * @brief Initialise the application
   *
   * @param argc command line arguments counter
   * @param argv command line arguments array
   * @return int error code, EXIT_SUCCESS (0) if success
   */
  int init(int argc, char **argv);

  /**
   * @brief run the application
   *
   */
  void run();

private:
  int parseArgs(int argc, char **argv);
};