#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "AppParams.h"
#include "Common.h"
#include "Layer.h"
#include "Manager.h"
#include "doctest.h"
#include <cstddef>
#include <filesystem>

using namespace sipai;

class MockRunnerVisitor : public RunnerVisitor {
public:
  mutable bool visitCalled = false;

  void visit(const TrainingData &trainingDataPairs,
             const TrainingData &validationDataPairs) const override {
    visitCalled = true;
  }
};

TEST_CASE("Testing the Manager class") {
  SUBCASE("Test constructor") {
    CHECK_NOTHROW({
      auto &manager = Manager::getInstance();
      MESSAGE(manager.app_params.title);
    });
  }

  SUBCASE("Test loadImage") {
    auto &manager = Manager::getInstance();
    auto &np = manager.network_params;
    np.input_size_x = 30;
    np.input_size_y = 30;
    size_t orig_x;
    size_t orig_y;
    auto image = manager.loadImage("../data/images/001a.png", orig_x, orig_y,
                                   np.input_size_x, np.input_size_y);
    CHECK(image.size() > 0);
    CHECK(orig_x > 0);
    CHECK(orig_y > 0);
    CHECK(image.size() == (np.input_size_x * np.input_size_y));
  }

  SUBCASE("Test saveImage") {
    auto &manager = Manager::getInstance();
    auto &np = manager.network_params;
    np.input_size_x = 30;
    np.input_size_y = 30;
    size_t orig_x;
    size_t orig_y;
    auto image = manager.loadImage("../data/images/001a.png", orig_x, orig_y,
                                   np.input_size_x, np.input_size_y);
    std::string tmpImage = "tmpImage.png";
    if (std::filesystem::exists(tmpImage)) {
      std::filesystem::remove(tmpImage);
    }
    CHECK(std::filesystem::exists(tmpImage) == false);
    manager.saveImage(tmpImage, image, np.input_size_x, np.input_size_y);
    CHECK(std::filesystem::exists(tmpImage) == true);
    auto image2 = manager.loadImage(tmpImage, orig_x, orig_y, np.input_size_x,
                                    np.input_size_y);
    CHECK(image2.size() == image.size());
    for (size_t i = 0; i < image2.size(); i++) {
      CHECK(image2.at(i).value == image.at(i).value);
    }
    std::filesystem::remove(tmpImage);
  }

  SUBCASE("Test initializeNetwork") {
    auto &manager = Manager::getInstance();
    manager.createNetwork();
    auto &np = manager.network_params;
    np.input_size_x = 2;
    np.input_size_y = 2;
    np.hidden_size_x = 3;
    np.hidden_size_y = 2;
    np.output_size_x = 3;
    np.output_size_y = 3;
    np.hiddens_count = 2;
    manager.initializeNetwork();

    const auto &network = manager.network;
    CHECK(network->layers.size() == (np.hiddens_count + 2));
    CHECK(network->layers.front()->layerType == LayerType::InputLayer);
    CHECK(network->layers.at(1)->layerType == LayerType::HiddenLayer);
    CHECK(network->layers.at(2)->layerType == LayerType::HiddenLayer);
    CHECK(network->layers.back()->layerType == LayerType::OutputLayer);

    const auto &inputLayer = network->layers.front();
    CHECK(inputLayer->neurons.size() == (np.input_size_x * np.input_size_y));
    CHECK(inputLayer->neurons.at(0).neighbors.size() == 2);

    const auto &hiddenLayer = network->layers.at(1);
    CHECK(hiddenLayer->neurons.size() == (np.hidden_size_x * np.hidden_size_y));
    CHECK(hiddenLayer->neurons.at(0).neighbors.size() == 2);

    const auto &outputLayer = network->layers.back();
    CHECK(outputLayer->neurons.size() == (np.output_size_x * np.output_size_y));
    CHECK(outputLayer->neurons.at(0).neighbors.size() == 2);

    manager.network.reset();
  }

  SUBCASE("Test import/export network") {
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
    ap.network_to_import = "tmpNetwork.json";
    ap.network_to_export = ap.network_to_import;
    std::string network_csv = "tmpNetwork.csv";

    if (std::filesystem::exists(ap.network_to_export)) {
      std::filesystem::remove(ap.network_to_export);
    }
    if (std::filesystem::exists(network_csv)) {
      std::filesystem::remove(network_csv);
    }

    // TEST EXPORT
    manager.createNetwork();
    manager.initializeNetwork();
    CHECK(std::filesystem::exists(ap.network_to_export) == false);
    manager.exportNetwork();
    CHECK(std::filesystem::exists(ap.network_to_export) == true);
    CHECK(std::filesystem::exists(network_csv) == true);
    manager.network.reset();

    // TEST IMPORT
    manager.importNetwork();
    auto &nn = manager.network;
    CHECK(nn->layers.size() == 3);
    CHECK(nn->layers.front()->layerType == LayerType::InputLayer);
    CHECK(nn->layers.at(1)->layerType == LayerType::HiddenLayer);
    CHECK(nn->layers.back()->layerType == LayerType::OutputLayer);
    CHECK(nn->layers.front()->neurons.size() == 4);
    CHECK(nn->layers.at(1)->neurons.size() == 6);
    CHECK(nn->layers.back()->neurons.size() == 9);
    CHECK(nn->layers.front()->neurons.at(0).neighbors.size() == 2);
    CHECK(nn->layers.back()->neurons.at(0).neighbors.size() == 2);

    std::filesystem::remove(ap.network_to_export);
    std::filesystem::remove(network_csv);
  }

  SUBCASE("Test loadTrainingData") {
    auto &manager = Manager::getInstance();
    manager.app_params.training_data_file = "images-test1.csv";
    const auto &data = manager.loadTrainingData();
    CHECK(data.size() == 10);
    for (auto &[source, target] : data) {
      CHECK(source.length() > 0);
      CHECK(target.length() > 0);
      CHECK_MESSAGE(std::filesystem::exists(source) == true, source);
      CHECK_MESSAGE(std::filesystem::exists(target) == true, target);
    }
  }

  SUBCASE("Test splitData") {
    auto &manager = Manager::getInstance();
    manager.app_params.training_data_file = "images-test1.csv";
    const auto &data = manager.loadTrainingData();
    const auto &split = manager.splitData(data, 0.8);
    CHECK(split.first.size() == 8);
    CHECK(split.second.size() == 2);
    for (auto &[source, target] : split.first) {
      CHECK(source.length() > 0);
      CHECK(target.length() > 0);
      CHECK_MESSAGE(std::filesystem::exists(source) == true, source);
      CHECK_MESSAGE(std::filesystem::exists(target) == true, target);
    }
    for (auto &[source, target] : split.second) {
      CHECK(source.length() > 0);
      CHECK(target.length() > 0);
      CHECK_MESSAGE(std::filesystem::exists(source) == true, source);
      CHECK_MESSAGE(std::filesystem::exists(target) == true, target);
    }
  }

  SUBCASE("Testing computeMSELoss method") {
    auto &manager = Manager::getInstance();
    std::vector<RGBA> outputImage(10);
    std::vector<RGBA> targetImage(10);

    RGBA pixel(0.1f, 0.2f, 0.3f, 0.4f);
    for (int i = 0; i < 10; ++i) {
      outputImage[i] = pixel;
      targetImage[i] = pixel;
    }

    float loss = manager.computeMSELoss(outputImage, targetImage);
    CHECK(loss == doctest::Approx(0.0));
  }

  SUBCASE("Testing computeMSELoss method with different images") {
    auto &manager = Manager::getInstance();
    std::vector<RGBA> outputImage(10);
    std::vector<RGBA> targetImage(10);

    RGBA pixel1(0.1f, 0.2f, 0.3f, 0.4f);
    RGBA pixel2(0.5f, 0.6f, 0.7f, 0.8f);
    for (int i = 0; i < 10; ++i) {
      outputImage[i] = pixel1;
      targetImage[i] = pixel2;
    }

    float loss = manager.computeMSELoss(outputImage, targetImage);
    CHECK(loss == doctest::Approx(0.16));
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

    auto &ap = manager.app_params;
    ap.training_data_file = "images-test1.csv";
    ap.max_epochs = 2;
    ap.run_mode = ERunMode::TrainingMonitored;
    ap.network_to_export = "tempNetwork.json";
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
    CHECK_NOTHROW(manager.run());
    CHECK(std::filesystem::exists(ap.network_to_export) == true);
    CHECK(std::filesystem::exists(network_csv) == true);
    std::filesystem::remove(ap.network_to_export);
    std::filesystem::remove(network_csv);
  }
}