
#include "SimpleLogger.h"
#include "include/SIPAI.h"
#include <cstdlib>
#include <exception>
#include <memory>

/**
 * @brief main function
 *
 * @param argc numbers of arguments
 * @param argv table of arguments
 *
 * @return int
 */
int main(int argc, char *argv[]) {
  try {
    auto sipai = std::make_unique<SIPAI>();
    int init = sipai->init(argc, argv);
    if (init == SIPAI::EXIT_HELP || init == SIPAI::EXIT_VERSION) {
      return EXIT_SUCCESS;
    } else if (init != EXIT_SUCCESS) {
      return init;
    }

    // TODO: sipai->run();

    return EXIT_SUCCESS;
  } catch (const std::exception &ex) {
    sipai::SimpleLogger::LOG_ERROR(ex.what());
    return EXIT_FAILURE;
  }
}