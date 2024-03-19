/**
 * \mainpage SIPAI
 * A Simple Image Processing Artificial Intelligence.
 * \author  Damien Balima (https://dams-labs.net)
 * \date    March 2024
 * \see https://github.com/obewan/SIPAI
 * \copyright
 *
 * [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]:
https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

- *Attribution*: you must give appropriate credit, provide a link to the
license, and indicate if changes were made. You may do so in any reasonable
manner, but not in any way that suggests the licensor endorses you or your use.
- *NonCommercial*: you may not use the material for commercial purposes.
- *ShareAlike*: if you remix, transform, or build upon the material, you must
distribute your contributions under the same license as the original.
- *No additional restrictions*: you may not apply legal terms or technological
measures that legally restrict others from doing anything the license permits.
 */

#include "include/SIPAI.h"
#include "AppParams.h"
#include "Manager.h"
#include "SimpleLogger.h"
#include "include/CLI11.hpp"
#include <cstdlib>
// for CLI11 doc, see https://github.com/CLIUtils/CLI11

using namespace sipai;

int SIPAI::init(int argc, char **argv) {
  if (argc == 1) {
    argv[1] = (char *)"-h"; // showing help by default
    argc++;
  }

  if (int init = parseArgs(argc, argv); init != EXIT_SUCCESS) {
    return init;
  }
  return EXIT_SUCCESS;
}

int SIPAI::parseArgs(int argc, char **argv) {
  auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;
  auto &network_params = manager.network_params;
  bool version = false;

  // valid a parent path, if there is a path, that include a futur filename,
  // useful for an export path (not like CLI::ExistingPath,
  // CLI::ExistingDirectory or CLI::ExistingFile)
  auto valid_path = [](auto filename) {
    if (std::filesystem::path p(filename);
        p.has_parent_path() && !std::filesystem::exists(p.parent_path())) {
      return std::string("Error: file not found.");
    }
    return std::string();
  };

  CLI::App app{app_params.title};
  app.add_option(
         "-i,--import_network", app_params.network_to_import,
         "Import a network model instead of creating a new one. This must be a "
         "valid model filepath (the JSON one), specifically a file generated "
         "by SIPAI (including its CSV part). Both of the JSON file and the CSV "
         "file of the model must exists. Indicate only the JSON file. If this "
         "option is used, there is no need to specify layer parameters as they "
         "are included in the model.")
      ->check(CLI::ExistingFile);
  app.add_option("-e,--export_network", app_params.network_to_export,
                 "Export the network model after training. This must be a "
                 "valid filepath. This will create two files, a JSON file that "
                 "include the metadata and a CSV file that include the neurons "
                 "weights. Both are necessary for an import.")
      ->check(valid_path);
  app.add_option("-t,--training_file", app_params.training_data_file,
                 "Specify the data file to be used for training and testing.")
      ->check(CLI::ExistingFile);
  app.add_option("--isx,--input_size_x", network_params.input_size_x,
                 "The X resolution for input neurons layer. This value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this width.")
      ->default_val(network_params.input_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--isy,--input_size_y", network_params.input_size_y,
                 "The Y resolution for input neurons layer. This value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this height.")
      ->default_val(network_params.input_size_y)
      ->check(CLI::PositiveNumber);
  app.add_flag("-v,--version", version, "Show current version.");

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