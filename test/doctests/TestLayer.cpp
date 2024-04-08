#include "Layer.h"
#include "LayerHidden.h"
#include "Manager.h"
#include "doctest.h"
#include <cstddef>
#include <memory>

using namespace sipai;

TEST_CASE("Testing Layer") {
  SUBCASE("Test updateWeights") {
    auto &manager = Manager::getInstance();
    bool disable_parallel = false;
    bool enable_parallel = true;
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
    manager.createOrImportNetwork();
    auto &outputLayer = manager.network->layers.back();
    CHECK(outputLayer->layerType == LayerType::LayerOutput);
    auto w1 = outputLayer->neurons.back().weights;

    for (auto &n : outputLayer->neurons) {
      n.error = {0.1, 0.2, 0.1, 0.3};
    }
    for (auto &n : outputLayer->previousLayer->neurons) {
      n.value = {0.1, 0.1, 0.5, 0.2};
    }
    CHECK_NOTHROW(outputLayer->updateWeights(
        manager.network_params.learning_rate, disable_parallel));
    auto w2 = outputLayer->neurons.back().weights;
    CHECK_FALSE(std::equal(w1.begin(), w1.end(), w2.begin()));

    for (auto &n : outputLayer->neurons) {
      n.error = {0.6, 0.1, -0.1, -0.3};
    }
    for (auto &n : outputLayer->previousLayer->neurons) {
      n.value = {0.7, 0.1, 0.1, 0.2};
    }
    CHECK_NOTHROW(outputLayer->updateWeights(
        manager.network_params.learning_rate, enable_parallel));
    auto w3 = outputLayer->neurons.back().weights;
    CHECK_FALSE(std::equal(w1.begin(), w1.end(), w3.begin()));
    CHECK_FALSE(std::equal(w2.begin(), w2.end(), w3.begin()));

    manager.network.reset();
  }
}