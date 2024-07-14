#include "Manager.h"
#include "NeuralNetwork.h"
#include "RunnerTrainingOpenCVVisitor.h"
#include "TrainingDataFactory.h"
#include "doctest.h"
#include "exception/RunnerVisitorException.h"
#include "exception/TrainingDataFactoryException.h"
#include <filesystem>
#include <memory>

using namespace sipai;

TEST_CASE("Testing RunnerTrainingOpenCVVisitor") {
  SUBCASE("Test exceptions") {
    RunnerTrainingOpenCVVisitor visitor;
    TrainingDataFactory::getInstance().clear();
    auto &manager = Manager::getInstance();

    // no network
    manager.network.reset();
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    // no data
    manager.network = std::make_unique<NeuralNetwork>();
    manager.app_params.training_data_file = "";
    manager.app_params.training_data_folder = "";
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    manager.network.reset();
  }

  SUBCASE("Test normal run") {
    RunnerTrainingOpenCVVisitor visitor;
    TrainingDataFactory::getInstance().clear();
    auto &manager = Manager::getInstance();

    auto &ap = manager.app_params;
    ap.training_data_file = "";
    ap.training_data_folder = "../data/images/target/";
    ap.max_epochs = 2;
    ap.run_mode = ERunMode::Training;
    ap.network_to_export = "tempNetwork.json";
    ap.network_to_import = "";
    ap.enable_vulkan = false;
    ap.random_loading = true;
    ap.verbose = true;
    ap.verbose_debug = true;
    std::string network_csv = "tempNetwork.csv";

    auto &np = manager.network_params;
    np.input_size_x = 2;
    np.input_size_y = 2;
    np.hidden_size_x = 3;
    np.hidden_size_y = 2;
    np.output_size_x = 3;
    np.output_size_y = 3;
    np.hiddens_count = 1;

    manager.createOrImportNetwork();
    CHECK_NOTHROW(visitor.visit());

    manager.network.reset();
    TrainingDataFactory::getInstance().clear();
  }
}