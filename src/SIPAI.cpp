#include "include/SIPAI.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "include/CLI11.hpp"
#include <cstdlib>

using namespace sipai;

int SIPAI::init(int argc, char **argv) {
  if (int init = parseArgs(argc, argv); init != EXIT_SUCCESS) {
    return init;
  }
  return EXIT_SUCCESS;
}

int SIPAI::parseArgs(int argc, char **argv) {
  const auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;
  bool version = false;

  CLI::App app{app_params.title};
  app.add_flag("-v,--version", version, "Show current version");

  // Parsing
  try {
    app.parse(argc, argv);
  } catch (const CLI::CallForHelp &e) {
    // This is returned when -h or --help is called
    app.exit(e);
    return EXIT_HELP;
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  }

  // Version special exit
  if (version) {
    const auto &logger = SimpleLogger::getInstance();
    logger.out(app_params.title, " v", app_params.version);
    logger.out("Copyright Damien Balima (https://dams-labs.net) 2024");
    return EXIT_VERSION;
  }

  return EXIT_SUCCESS;
}

void SIPAI::run() { Manager::getInstance().run(); }