#include <cstddef>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "Layer.h"
#include "Manager.h"
#include "doctest.h"
#include <filesystem>

using namespace sipai;

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
}