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

#include "SIPAI.h"
#include "CLI11.hpp"
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
      return std::string("Error: invalid path.");
    }
    return std::string();
  };

  // Optional: using a config file, TOML format
  // see https://github.com/CLIUtils/CLI11?tab=readme-ov-file#configuration-file
  // see https://cliutils.github.io/CLI11/book/chapters/config.html
  // see https://toml.io/en/
  app.set_config("-c, --config", "sipai.toml",
                 "optional: read a specific config file or a default "
                 "sipai.toml file, using TOML format and same options names.\n"
                 "Example of sipai.toml file:\n"
                 "mode = \"Training\"\n"
                 "input_size_x = 20\n"
                 "input_size_y = 20");

  // Single hyphen
  app.add_option(
         "-m, --mode", app_params.run_mode,
         "Select the running mode:\n  - Enhancer:This mode uses an "
         "input image to generate its enhanced image (default).\n    The "
         "enhancer mode requires a neural network that has been imported and "
         "trained for enhancement (be sure that the model has good testing "
         "results).\n  - Testing: Test an imported neural network without "
         "training.\n "
         " - Training: Train and test at each epoch, showing training "
         "progress.")
      ->default_val(app_params.run_mode)
      ->transform(CLI::CheckedTransformer(mode_map, CLI::ignore_case));
  app.add_flag("-v,--version", version, "Show current version.");
  app.add_flag("-V,--verbose", app_params.verbose,
               "Enables verbose logs. Can be useful during long epoch "
               "treatment, to check it is not pending.");
  app.add_flag(
      "-D, --verbose_debug", app_params.verbose_debug,
      "Enables verbose debug-level logs. Can be used with verbose logs for "
      "even more detailed logs, but at the cost of some performance.");
  app.add_flag("-G, --vulkan_debug", app_params.vulkan_debug,
               "Enables vulkan debug mode, to debug with RenderDoc and other "
               "tools. Do not enable in production.");

  // Double hyphen
  app.add_option(
         "--in,--import_network", app_params.network_to_import,
         "Import a neural network model instead of creating a new one. This "
         "must be a valid model filepath (the JSON one), \nspecifically a file "
         "generated by SIPAI. Ex: --in myModel.json\n"
         "Both of the JSON file and the CSV file of the model must exist."
         "Indicate only the JSON file. \nIf this "
         "option is used, there is no need to specify layer parameters as they "
         "are included in the model.")
      ->check(CLI::ExistingFile);
  app.add_option(
         "--en,--export_network", app_params.network_to_export,
         "Export the neural network model after training.\nThis must be a "
         "valid filepath, with a json extension. Ex: --en myModel.json\n"
         "This will create two files, a JSON file that "
         "includes the metadata and a CSV file that includes the neurons "
         "weights. \nBoth are necessary for an import. ")
      ->check(valid_path);
  app.add_option(
         "--if,--input_file", app_params.input_file,
         "The path to the input image file to be enhanced.\nThis option "
         "is used in conjunction with the Enhancer mode.\nThe specified "
         "file must exist. Currently supported image format: "
         "\n.bmp, .dib, .jpeg, .jpg, .jpe, .jp2, .png, .webp, .pbm, .pgm,  "
         "\n.ppm, .pxm, .pnm, .pfm, .sr, .ras, .tiff, .tif, .exr, .hdr, .pic")
      ->check(CLI::ExistingFile);
  app.add_option(
         "--of,--output_file", app_params.output_file,
         "The path where the enhanced output image will be saved.\nThis option "
         "is used in conjunction with the Enhancer mode.\nThe path must be "
         "valid, and the application must have write permissions to the "
         "specified location."
         "\nSee the input_file parameter for the supported image formats.")
      ->check(valid_path);
  app.add_option("--os,--output_scale", app_params.output_scale,
                 "The scale of the output image.\nThis option "
                 "is used in conjunction with the Enhancer mode.")
      ->default_val(app_params.output_scale)
      ->check(CLI::PositiveNumber)
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "--tfi,--training_file", app_params.training_data_file,
         "Specify the data file to be used for training and testing.\n"
         "It must be a valid CSV file with two columns, \nwhere the first "
         "column contains the input file path, and the second "
         "column contains the corresponding target file path. No headers."
         "\nSee the input_file parameter for the supported image formats.")
      ->check(CLI::ExistingFile);
  app.add_option(
         "--tfo,--training_folder", app_params.training_data_folder,
         "Specify the directory containing target images for training "
         "and testing. "
         "\nInput files will be automatically generated by reducing the "
         "resolution of the target images "
         "according to the 'training_reduce_factor' parameter. "
         "\nNote: When using this option, there's no need for a separate "
         "training CSV file or pre-prepared input files."
         "\nSee the input_file parameter for the supported image formats.")
      ->check(CLI::ExistingDirectory);
  app.add_option(
         "--trf,--training_reduce_factor", app_params.training_reduce_factor,
         "Specify the factor by which to reduce the resolution of "
         "training images in the specified training data folder. "
         "\nThis reduced-resolution image will be used as the input for "
         "training.")
      ->default_val(app_params.training_reduce_factor)
      ->check(CLI::PositiveNumber);
  app.add_option("--isx,--input_size_x", network_params.input_size_x,
                 "The X resolution for input layer.\nThis value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this width.")
      ->default_val(network_params.input_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--isy,--input_size_y", network_params.input_size_y,
                 "The Y resolution for input layer.\nThis value should "
                 "not be too large to avoid performance degradation. Incoming "
                 "images will be resized to this height.")
      ->default_val(network_params.input_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option(
         "--hsx,--hidden_size_x", network_params.hidden_size_x,
         "The X resolution for any hidden layer.\nThis value should "
         "not be too large to avoid performance degradation, and should "
         "be around the input size X and the output size X.")
      ->default_val(network_params.hidden_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--hsy,--hidden_size_y", network_params.hidden_size_y,
                 "The Y resolution for any hidden layer.\nThis value should "
                 "not be too large to avoid performance degradation, and "
                 "should be around the input size Y and the output size Y.")
      ->default_val(network_params.hidden_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option("--osx,--output_size_x", network_params.output_size_x,
                 "The X resolution for the output layer.\nThis value should "
                 "not be too large to avoid performance degradation. Target "
                 "images will be resized to this width.")
      ->default_val(network_params.output_size_x)
      ->check(CLI::PositiveNumber);
  app.add_option("--osy,--output_size_y", network_params.output_size_y,
                 "The Y resolution for the output layer.\nThis value should "
                 "not be too large to avoid performance degradation. Target "
                 "images will be resized to this height.")
      ->default_val(network_params.output_size_y)
      ->check(CLI::PositiveNumber);
  app.add_option("--hl,--hiddens_layers", network_params.hiddens_count,
                 "The number of hidden layers.")
      ->default_val(network_params.hiddens_count)
      ->check(CLI::NonNegativeNumber);
  app.add_option("--ep,--epochs", app_params.max_epochs,
                 "The maximum number of epochs to run during training. For no "
                 "maximum, indicate " +
                     std::to_string(NO_MAX_EPOCHS))
      ->default_val(app_params.max_epochs)
      ->check(CLI::NonNegativeNumber);
  app.add_option(
         "--epwi,--epochs_without_improvement",
         app_params.max_epochs_without_improvement,
         "The maximum number of epochs without improvement during a training "
         "after which the training will stop.")
      ->default_val(app_params.max_epochs_without_improvement)
      ->check(CLI::NonNegativeNumber);
  app.add_option("--tsr, --training_split_ratio",
                 app_params.training_split_ratio,
                 "The training ratio of the file to switch between data for "
                 "training and data for testing, should be around 0.7.")
      ->default_val(app_params.training_split_ratio)
      ->check(CLI::Range(0.0f, 1.0f))
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "--lr, --learning_rate", network_params.learning_rate,
         "The learning rate for training the neural network.\nThis is a "
         "crucial hyperparameter that controls how much the weights of the "
         "network will change in response to the error at each step of the "
         "learning process. \nA smaller learning rate could make the learning "
         "process slower but more precise, \nwhile a larger learning rate "
         "could make learning faster but risk overshooting the optimal "
         "solution.")
      ->default_val(network_params.learning_rate)
      ->check(CLI::Range(app_params.learning_rate_min,
                         app_params.learning_rate_max))
      ->check(CLI::TypeValidator<float>());
  app.add_flag(
      "--alr, --adaptive_learning_rate", network_params.adaptive_learning_rate,
      "This flag enables the adaptive adjustment of the learning rate "
      "based on the validation loss. "
      "\nWhen enabled, the learning rate will decrease if the "
      "validation loss increases. "
      "\nThis dynamic adjustment can potentially improve the efficiency "
      "and effectiveness of the learning process by preventing "
      "overshooting of the optimal solution.");
  app.add_flag(
      "--alri, --adaptive_learning_rate_increase",
      network_params.enable_adaptive_increase,
      "This flag enables the adaptive increase of the learning rate when the "
      "validation loss decreases. "
      "\nThis dynamic adjustment can potentially speed up the learning process "
      "when the model is improving. "
      "\nNote: This option requires the 'adaptive_learning_rate' option to be "
      "enabled, as it is a further enhancement of that feature.");
  app.add_option(
         "--alrf, --adaptive_learning_rate_factor",
         network_params.adaptive_learning_rate_factor,
         "This option sets the factor by which the learning rate is adjusted "
         "during the adaptive learning rate process. "
         "\nIt inversely affects the magnitude of learning rate adjustments. "
         "A smaller factor will result in larger changes to the learning rate, "
         "potentially leading to faster adaptation. "
         "\nHowever, be aware that a smaller factor can also increase the risk "
         "of instability in the learning process, as it may cause the learning "
         "rate to change too rapidly. "
         "\nChoose a value that balances learning speed and stability.")
      ->default_val(network_params.adaptive_learning_rate_factor)
      ->check(CLI::Range(0.0001f, 1.0f))
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "--temin, --training_error_min", network_params.error_min,
         "Specifies the minimum error value during the backward propagation "
         "phase of training. "
         "\nSetting this to a very low value may cause the gradients to "
         "explode, "
         "leading to instability in the learning process. "
         "\nChoose a value that balances learning efficiency and stability.")
      ->default_val(network_params.error_min)
      ->check(CLI::Range(-100000.0f, 0.0f))
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "--temax, --training_error_max", network_params.error_max,
         "Specifies the maximum error value during the backward propagation "
         "phase of training. "
         "\nSetting this to a very high value may cause the gradients to "
         "vanish, "
         "slowing down the learning process or causing it to get stuck. "
         "\nChoose a value that balances learning efficiency and stability.")
      ->default_val(network_params.error_max)
      ->check(CLI::Range(0.0f, 100000.0f))
      ->check(CLI::TypeValidator<float>());
  app.add_option(
         "--haf,--hidden_activation_function",
         network_params.hidden_activation_function,
         "Select the hidden neurons activation function:\n  - ELU: "
         "Exponential "
         "Linear Units, require an hidden_activation_alpha parameter.\n  - "
         "LReLU: Leaky ReLU.\n  - PReLU: Parametric ReLU, require an "
         "hidden_activation_alpha_parameter.\n  - ReLU: Rectified Linear "
         "Unit (default).\n  - Sigmoid.\n  - Tanh: Hyperbolic Tangent")
      ->default_val(network_params.hidden_activation_function)
      ->transform(CLI::CheckedTransformer(activation_map, CLI::ignore_case));
  app.add_option(
         "--oaf,--output_activation_function",
         network_params.output_activation_function,
         "Select the output neurons activation function:\n  - ELU: Exponential "
         "Linear Units, require an hidden_activation_alpha parameter.\n  - "
         "LReLU: Leaky ReLU.\n  - PReLU: Parametric ReLU, require an "
         "hidden_activation_alpha_parameter.\n  - ReLU: Rectified Linear "
         "Unit (default).\n  - Sigmoid.\n  - Tanh: Hyperbolic Tangent")
      ->default_val(network_params.output_activation_function)
      ->transform(CLI::CheckedTransformer(activation_map, CLI::ignore_case));
  app.add_option("--haa, --hidden_activation_alpha",
                 network_params.hidden_activation_alpha,
                 "The alpha parameter value for ELU and PReLU activation "
                 "functions on hidden layer(s).")
      ->default_val(network_params.hidden_activation_alpha)
      ->check(CLI::Range(-100.0f, 100.0f));
  app.add_option("--oaa, --output_activation_alpha",
                 network_params.output_activation_alpha,
                 "The alpha parameter value for ELU and PReLU activation "
                 "functions on output layer.")
      ->default_val(network_params.output_activation_alpha)
      ->check(CLI::Range(-100.0f, 100.0f));
  app.add_option(
         "--eas, --epoch_auto_save", app_params.epoch_autosave,
         "The frequency (in number of epochs) at which the neural "
         "network will be saved and exported to the file path specified "
         "by the export_network option.")
      ->default_val(app_params.epoch_autosave)
      ->check(CLI::PositiveNumber);
  app.add_option(
         "--is, --image_split", app_params.image_split,
         "Split the training image into smaller parts, in x and y, that will "
         "fit better smaller neural network input layer resolution,\nand so "
         "should improve the final result, which will be reconstitued from the "
         "smaller parts.\nFor example, a split of 2 will result in 4 smaller "
         "parts, and a split of 3 will result in 9 smaller parts.\nTo avoid "
         "padding, make sure the split number is a multiple of the image width "
         "and height.\nAlso, be aware that using a split will increase the "
         "training time.\nFor no image_split, indicate 1 or " +
             std::to_string(NO_IMAGE_SPLIT))
      ->default_val(app_params.image_split)
      ->check(CLI::NonNegativeNumber);
  app.add_flag(
      "--pad, --padding", app_params.enable_padding,
      "Enable this flag to add padding to the image if the split factor is not "
      "a multiple of the image width and height.\nPadding helps to avoid image "
      "ratio deformation when splitting the image into smaller "
      "parts.\nHowever, "
      "it adds extra pixels to the image that are not part of the original "
      "content.\nThese extra pixels are filled with a default color (black in "
      "this case).");
  app.add_flag(
      "--rl,--random_loading", app_params.random_loading,
      "This flag will activate a random loading of all images, for a better "
      "training if the images list or folder is not already randomized.");
  app.add_flag("--bl,--bulk_loading", app_params.bulk_loading,
               "This flag will activate a bulk loading of all images, "
               "instead of loading and unloading them, resulting of training "
               "speed but at the cost of more memory,\n"
               "depending on the images total count and size.");
  app.add_flag(
      "--par,--parallelism", app_params.enable_parallel,
      "Enables CPU parallel processing for neural network computations. ");
  app.add_flag("--vul,--vulkan", app_params.enable_vulkan,
               "Enables GPU acceleration by leveraging Vulkan "
               "for processing (experimental). "
               "\nWhen enabled, it utilizes the GPU to speed up computations, "
               "potentially "
               "leading to significant performance improvements for "
               "large-scale operations."
               "\nPlease note that enabling "
               "this flag will override the 'parallelism' setting, as the "
               "parallel processing will be handled by the Vulkan API instead "
               "of the CPU, except if Vulkan failed to initialize.");
}

void SIPAI::run() {
  Manager::getInstance()
      .showHeader()
      .createOrImportNetwork()
      .showParameters()
      .run();
}