#include "Layer.h"
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "Manager.h"
#include "doctest.h"

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
    auto image = manager.loadImage("../data/images/001a.png");
    CHECK(image.size() > 0);
    CHECK(image.size() == (manager.network_params.input_size_x *
                           manager.network_params.input_size_y));
  }

  SUBCASE("Test initializeNetwork") {
    auto &manager = Manager::getInstance();
    manager.createNetwork();
    auto &network_params = manager.network_params;
    network_params.input_size_x = 2;
    network_params.input_size_y = 2;
    network_params.hidden_size_x = 3;
    network_params.hidden_size_y = 2;
    network_params.output_size_x = 3;
    network_params.output_size_y = 3;
    network_params.hiddens_count = 2;
    manager.initializeNetwork();

    const auto &network = manager.network;
    CHECK(network->layers.size() == (network_params.hiddens_count + 2));
    CHECK(network->layers.front()->layerType == LayerType::InputLayer);
    CHECK(network->layers.at(1)->layerType == LayerType::HiddenLayer);
    CHECK(network->layers.at(2)->layerType == LayerType::HiddenLayer);
    CHECK(network->layers.back()->layerType == LayerType::OutputLayer);

    const auto &inputLayer = network->layers.front();
    CHECK(inputLayer->neurons.size() ==
          (network_params.input_size_x * network_params.input_size_y));
    CHECK(inputLayer->neurons.at(0).neighbors.size() == 2);

    const auto &hiddenLayer = network->layers.at(1);
    CHECK(hiddenLayer->neurons.size() ==
          (network_params.hidden_size_x * network_params.hidden_size_y));
    CHECK(hiddenLayer->neurons.at(0).neighbors.size() == 2);

    const auto &outputLayer = network->layers.back();
    CHECK(outputLayer->neurons.size() ==
          (network_params.output_size_x * network_params.output_size_y));
    CHECK(outputLayer->neurons.at(0).neighbors.size() == 2);

    manager.network.reset();
  }
}