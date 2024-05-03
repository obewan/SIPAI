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
    auto w1 = outputLayer->neurons.back().back().weights.clone();

    outputLayer->errors =
        cv::Mat(3, 3, CV_32FC4, cv::Vec4f{1.5, 3.2, 2.1, 5.3});
    outputLayer->previousLayer->errors =
        cv::Mat(3, 2, CV_32FC4, cv::Vec4f{5.1, 1.1, -5.5, 2.2});
    CHECK_NOTHROW(
        outputLayer->updateWeights(manager.network_params.learning_rate));

    // get the weights after update
    auto w2 = outputLayer->neurons.back().back().weights.clone();

    CHECK(w1.type() == w2.type());
    CHECK(w1.size() == w2.size());

    std::cout << "weights before:\n" << w1 << std::endl;
    std::cout << "weights after:\n" << w2 << std::endl;

    bool isEq = true;
    for (int y = 0; y < w1.rows; ++y) {
      for (int x = 0; x < w1.cols; ++x) {
        for (int c = 0; c < 4; ++c) {
          if (std::abs(w1.at<cv::Vec4f>(y, x)[c] - w2.at<cv::Vec4f>(y, x)[c]) >=
              std::numeric_limits<float>::epsilon()) {
            isEq = false;
            break;
          }
        }
        if (!isEq) {
          break;
        }
      }
      if (!isEq) {
        break;
      }
    }
    CHECK(isEq == false);
    manager.network.reset();
  }
}