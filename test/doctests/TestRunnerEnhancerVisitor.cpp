#include "Manager.h"
#include "NeuralNetwork.h"
#include "RunnerEnhancerVisitor.h"
#include "doctest.h"
#include "exception/RunnerVisitorException.h"
#include <filesystem>
#include <memory>

using namespace sipai;

TEST_CASE("Testing RunnerEnhancerVisitor") {

  SUBCASE("Test exceptions") {
    RunnerEnhancerVisitor visitor;
    auto &manager = Manager::getInstance();
    manager.network.reset();
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    manager.network = std::make_unique<NeuralNetwork>();
    manager.app_params.input_file = "";
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    manager.app_params.input_file = "../data/images/input/001a.png";
    manager.app_params.output_file = "";
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);
  }
  SUBCASE("Test visit success") {
    RunnerEnhancerVisitor visitor;
    auto &manager = Manager::getInstance();
    manager.network.reset();
    manager.network_params = {
        .input_size_x = 2,
        .input_size_y = 2,
        .hidden_size_x = 3,
        .hidden_size_y = 2,
        .output_size_x = 3,
        .output_size_y = 3,
        .hiddens_count = 2,
    };
    manager.app_params.network_to_import = "";
    manager.app_params.enable_vulkan = false;
    manager.createOrImportNetwork();
    manager.app_params.input_file = "../data/images/input/001a.png";
    manager.app_params.output_file = "../data/images/output/001a_test.png";
    if (std::filesystem::exists(manager.app_params.output_file)) {
      std::filesystem::remove(manager.app_params.output_file);
    }

    CHECK_NOTHROW(visitor.visit());
    CHECK(std::filesystem::exists(manager.app_params.output_file));

    if (std::filesystem::exists(manager.app_params.output_file)) {
      std::filesystem::remove(manager.app_params.output_file);
    }
    manager.network.reset();
  }
}