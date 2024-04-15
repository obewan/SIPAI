#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "AppParams.h"
#include "Common.h"
#include "ImageHelper.h"
#include "Layer.h"
#include "Manager.h"
#include "doctest.h"
#include <cstddef>
#include <filesystem>

using namespace sipai;

class MockRunnerVisitor : public RunnerVisitor {
public:
  mutable bool visitCalled = false;

  void visit() const override { visitCalled = true; }
};

TEST_CASE("Testing the Manager class") {
  SUBCASE("Test constructor") {
    CHECK_NOTHROW({
      auto &manager = Manager::getInstance();
      MESSAGE(manager.app_params.title);
    });
  }

  SUBCASE("Test initializeNetwork") {
    auto &manager = Manager::getInstance();
    auto &np = manager.network_params;
    np.input_size_x = 2;
    np.input_size_y = 2;
    np.hidden_size_x = 3;
    np.hidden_size_y = 2;
    np.output_size_x = 3;
    np.output_size_y = 3;
    np.hiddens_count = 2;
    manager.app_params.network_to_import = "";
    manager.network.reset();
    manager.createOrImportNetwork();

    const auto &network = manager.network;
    CHECK(network->layers.size() == (np.hiddens_count + 2));
    CHECK(network->layers.front()->layerType == LayerType::LayerInput);
    CHECK(network->layers.at(1)->layerType == LayerType::LayerHidden);
    CHECK(network->layers.at(2)->layerType == LayerType::LayerHidden);
    CHECK(network->layers.back()->layerType == LayerType::LayerOutput);

    const auto &inputLayer = network->layers.front();
    CHECK(inputLayer->neurons.size() == (np.input_size_x * np.input_size_y));
    CHECK(inputLayer->neurons.at(0).neighbors.size() == 0);

    const auto &hiddenLayer = network->layers.at(1);
    CHECK(hiddenLayer->neurons.size() == (np.hidden_size_x * np.hidden_size_y));
    CHECK(hiddenLayer->neurons.at(0).neighbors.size() == 2);

    const auto &outputLayer = network->layers.back();
    CHECK(outputLayer->neurons.size() == (np.output_size_x * np.output_size_y));
    CHECK(outputLayer->neurons.at(0).neighbors.size() == 2);

    manager.network.reset();
  }

  SUBCASE("Test import/export network") {
    const float eps = 1e-6f; // epsilon for float testing
    auto &manager = Manager::getInstance();
    auto &ap = manager.app_params;
    auto &np = manager.network_params;
    np.input_size_x = 2;
    np.input_size_y = 2;
    np.hidden_size_x = 3;
    np.hidden_size_y = 2;
    np.output_size_x = 3;
    np.output_size_y = 3;
    np.hiddens_count = 1;
    np.learning_rate = 0.02f;
    np.adaptive_learning_rate = true;
    np.adaptive_learning_rate_factor = 0.123;
    ap.network_to_import = "";
    ap.network_to_export = "tmpNetwork.json";
    std::string network_csv = "tmpNetwork.csv";

    // TEST CREATE AND EXPORT
    ap.network_to_import = "";
    if (std::filesystem::exists(ap.network_to_export)) {
      std::filesystem::remove(ap.network_to_export);
    }
    if (std::filesystem::exists(network_csv)) {
      std::filesystem::remove(network_csv);
    }
    // CREATE
    manager.createOrImportNetwork();
    // EXPORT
    CHECK_FALSE(std::filesystem::exists(ap.network_to_export));
    manager.exportNetwork();
    CHECK(std::filesystem::exists(ap.network_to_export));
    CHECK(std::filesystem::exists(network_csv));

    // TEST IMPORT
    manager.network.reset();
    manager.network_params = {};
    CHECK(np.input_size_x != 2);
    ap.network_to_import = "tmpNetwork.json";
    manager.createOrImportNetwork();
    CHECK(np.input_size_x == 2);
    CHECK(np.input_size_y == 2);
    CHECK(np.hidden_size_x == 3);
    CHECK(np.hidden_size_y == 2);
    CHECK(np.output_size_x == 3);
    CHECK(np.output_size_y == 3);
    CHECK(np.hiddens_count == 1);
    CHECK(np.learning_rate == doctest::Approx(0.02f).epsilon(eps));
    CHECK(np.adaptive_learning_rate == true);
    CHECK(np.adaptive_learning_rate_factor ==
          doctest::Approx(0.123).epsilon(eps));
    auto &nn = manager.network;
    CHECK(nn->layers.size() == 3);
    CHECK(nn->layers.front()->layerType == LayerType::LayerInput);
    CHECK(nn->layers.at(1)->layerType == LayerType::LayerHidden);
    CHECK(nn->layers.back()->layerType == LayerType::LayerOutput);
    CHECK(nn->layers.front()->neurons.size() == 4);
    CHECK(nn->layers.at(1)->neurons.size() == 6);
    CHECK(nn->layers.back()->neurons.size() == 9);
    CHECK(nn->layers.front()->neurons.at(0).neighbors.size() == 0);
    CHECK(nn->layers.back()->neurons.at(0).neighbors.size() == 2);

    std::filesystem::remove(ap.network_to_export);
    std::filesystem::remove(network_csv);
    manager.network.reset();
  }

  SUBCASE("Testing runWithVisitor") {
    auto &manager = Manager::getInstance();
    manager.app_params.training_data_file = "images-test1.csv";
    MockRunnerVisitor visitor;

    manager.runWithVisitor(visitor);

    CHECK(visitor.visitCalled == true);
  }

  SUBCASE("Testing run") {
    auto &manager = Manager::getInstance();
    manager.network.reset();

    auto &ap = manager.app_params;
    ap.training_data_file = "images-test1.csv";
    ap.max_epochs = 2;
    ap.run_mode = ERunMode::TrainingMonitored;
    ap.network_to_export = "tempNetwork.json";
    ap.network_to_import = "";
    std::string network_csv = "tempNetwork.csv";

    auto &np = manager.network_params;
    np.input_size_x = 2;
    np.input_size_y = 2;
    np.hidden_size_x = 3;
    np.hidden_size_y = 2;
    np.output_size_x = 3;
    np.output_size_y = 3;
    np.hiddens_count = 1;

    if (std::filesystem::exists(ap.network_to_export)) {
      std::filesystem::remove(ap.network_to_export);
    }
    if (std::filesystem::exists(network_csv)) {
      std::filesystem::remove(network_csv);
    }
    CHECK(std::filesystem::exists(ap.training_data_file));
    CHECK_NOTHROW(manager.run());
    CHECK(std::filesystem::exists(ap.network_to_export));
    CHECK(std::filesystem::exists(network_csv));
    std::filesystem::remove(ap.network_to_export);
    std::filesystem::remove(network_csv);
    manager.network.reset();
  }
}