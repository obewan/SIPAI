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
#include "Manager.h"
#include "SimpleLogger.h"
#include <cstdlib>
#include <string>
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

  CLI::App app{app_params.title};
  addOptions(app, app_params, network_params, version);

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

void SIPAI::addOptions(CLI::App &app, AppParams &app_params,
                       NeuralNetworkParams &network_params, bool &version) {
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

  app.add_option(
         "-i,--import_network", app_params.network_to_import,
         "Import a neural network model instead of creating a new one. This "
         "must be a "
         "valid model filepath (the JSON one), specifically a file generated "
         "by SIPAI (including its CSV part). Both of the JSON file and the CSV "
         "file of the model must exist. Indicate only the JSON file. If this "
         "option is used, there is no need to specify layer parameters as they "
         "are included in the model.")
      ->check(CLI::ExistingFile);
  app.add_option(
         "-e,--export_network", app_params.network_to_export,
         "Export the neural network model after training. This must be a "
         "valid filepath. This will create two files, a JSON file that "
         "includes the metadata and a CSV file that includes the neurons "
         "weights. Both are necessary for an import.")
      ->check(valid_path);
  app.add_option(
         "-t,--training_file", app_params.training_data_file,
         "Specify the data file to be used for training and testing. "
         "It must be a valid CSV file with two columns, where the first "
         "column contains the input file path, and the second "
         "column contains the corresponding target file path. No headers.")
      ->check(CLI::ExistingFile);
  app.add_option("--isx,--input_size_x", network_params.input_size_x,
                 "The X resolution for input layer. This value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this width.")
      ->default_val(network_params.input_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--isy,--input_size_y", network_params.input_size_y,
                 "The Y resolution for input layer. This value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this height.")
      ->default_val(network_params.input_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option(
         "--hsx,--hidden_size_x", network_params.hidden_size_x,
         "The X resolution for any hidden layer. This value should "
         "not be too large to avoid performance degradation, and should "
         "be around the input size X and the output size X.")
      ->default_val(network_params.hidden_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--hsy,--hidden_size_y", network_params.hidden_size_y,
                 "The Y resolution for any hidden layer. This value should "
                 "not be too large to avoid performance degradation, and "
                 "should be around the input size Y and the output size Y.")
      ->default_val(network_params.hidden_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option("--osx,--output_size_x", network_params.output_size_x,
                 "The X resolution for the output layer. This value should "
                 "not be too large to avoid performance degradation. Target "
                 "images will be resized to this width.")
      ->default_val(network_params.output_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--osy,--output_size_y", network_params.output_size_y,
                 "The Y resolution for the output layer. This value should "
                 "not be too large to avoid performance degradation. Target "
                 "images will be resized to this height.")
      ->default_val(network_params.output_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option("--hl,--hiddens_layers", network_params.hiddens_count,
                 "The number of hidden layers.")
      ->default_val(network_params.hiddens_count)
      ->check(CLI::NonNegativeNumber);
  app.add_option("-p,--epochs", app_params.max_epochs,
                 "The maximum number of epochs to run during training. For no "
                 "maximum, indicate " +
                     std::to_string(NOMAX_EPOCHS))
      ->default_val(app_params.max_epochs)
      ->check(CLI::NonNegativeNumber);
  app.add_option(
         "--pwi,--epochs_without_improvement",
         app_params.max_epochs_without_improvement,
         "The maximum number of epochs without improvement during a training "
         "after which the training will stop.")
      ->default_val(app_params.max_epochs_without_improvement)
      ->check(CLI::NonNegativeNumber);
  app.add_option("-r, --split_ratio", app_params.split_ratio,
                 "The training ratio of the file to switch between data for "
                 "training and data for testing, should be around 0.7.")
      ->default_val(app_params.split_ratio)
      ->check(CLI::Range(0.0f, 1.0f))
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "-m, --mode", app_params.run_mode,
         "Select the running mode:\n  - Enhancer:This mode uses an "
         "input image "
         "to generate its enhanced image. (default)\n    The enhancer mode "
         "requires a neural network that has been imported and trained for "
         "enhancement (be sure that the model has good testing "
         "results).\n  - Testing: Test an imported neural network without "
         "training.\n  - Training: Train the neural network without testing.\n "
         " - TrainingMonitored: Train and test at each epoch while monitoring "
         "the progress. Be aware that this is slower and will use more "
         "memory.")
      ->default_val(app_params.run_mode)
      ->transform(CLI::CheckedTransformer(mode_map, CLI::ignore_case));
  app.add_flag("-v,--version", version, "Show current version.");
}

void SIPAI::run() { Manager::getInstance().run(); }