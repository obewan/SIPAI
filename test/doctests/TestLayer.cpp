#include "Layer.h"
#include "LayerHidden.h"
#include "Manager.h"
#include "doctest.h"
#include <cstddef>
#include <memory>

using namespace sipai;

TEST_CASE("Testing Layer")
{
  SUBCASE("Test updateWeights")
  {
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
    manager.network_params.learning_rate = 0.5;
    manager.app_params.network_to_import = "";
    manager.createOrImportNetwork();
    auto &outputLayer = manager.network->layers.back();
    CHECK(outputLayer->layerType == LayerType::LayerOutput);
    CHECK(outputLayer->neurons.size() > 0);
    CHECK(outputLayer->neurons.back().size() > 0);
    CHECK(outputLayer->previousLayer != nullptr);
    cv::randu(outputLayer->previousLayer->values, 0, 1);

    // get the weights before update
    auto oldWeights = outputLayer->neurons.back().back().weights.clone();

    outputLayer->errors =
        cv::Mat(3, 3, CV_32FC4, cv::Vec4f{1.5f, 3.2f, 2.1f, 5.3f});
    outputLayer->previousLayer->errors =
        cv::Mat(3, 2, CV_32FC4, cv::Vec4f{5.1f, 1.1f, -5.5f, 2.2f});
    CHECK_NOTHROW(
        outputLayer->updateWeights(manager.network_params.learning_rate));

    // get the weights after update
    auto newWeights = outputLayer->neurons.back().back().weights.clone();

    CHECK(oldWeights.type() == newWeights.type());
    CHECK(oldWeights.size() == newWeights.size());

    std::cout << "weights before:\n"
              << oldWeights << std::endl;
    std::cout << "weights after:\n"
              << newWeights << std::endl;

    //  Check if oldWeights and newWeights are not equals
    cv::Mat diff = cv::abs(oldWeights - newWeights);
    double norm = cv::norm(diff, cv::NORM_L1);
    bool isEq = norm < std::numeric_limits<float>::epsilon() * diff.total();
    CHECK(isEq == false);

    manager.network.reset();
  }
}