#include "ActivationFunctions.h"
#include "LayerHidden.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkBuilder.h"
#include "NeuralNetworkParams.h"
#include "doctest.h"
#include "exception/NeuralNetworkException.h"
#include <cstddef>
#include <memory>
#include <opencv2/core/matx.hpp>

using namespace sipai;

TEST_CASE("Testing the Activation Functions") {

  auto network = std::make_unique<NeuralNetwork>();
  auto hlayer = new LayerHidden();
  network->layers.push_back(hlayer);
  NeuralNetworkBuilder builder;
  builder.with(network);
  NeuralNetworkParams networkParams;
  const float eps = 1e-6f; // epsilon for float testing
  float alpha = 0.1f;
  struct hasActivationFunctions {
    bool hasElu = false;
    bool hasLRelu = false;
    bool hasPRelu = false;
    bool hasRelu = false;
    bool hasSigm = false;
    bool hasTanh = false;
  };
  hasActivationFunctions hasAF;

  auto testELU = [&eps](const Layer &lay) {
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0, 1.0, 1.0, 1.0});
    CHECK(lay.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f})[0] ==
          doctest::Approx(0).epsilon(eps));
    CHECK(lay.activationFunction({0.0f, 0.0f, 0.0f, 0.0f}) ==
          cv::Vec4f{0.0f, 0.0f, 0.0f, 0.0f});
    CHECK(lay.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0, 1.0, 1.0, 1.0});
    CHECK(lay.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})[0] ==
          doctest::Approx(0.0367879f).epsilon(eps));
  };

  auto testLReLU = [&eps](const Layer &lay) {
    // Positive values pass through (clamped to [0,1])
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    // Negative values are scaled by 0.01, then clamped to [0,1] -> 0
    CHECK(lay.activationFunction({-0.5f, -0.5f, -0.5f, -0.5f})[0] ==
          doctest::Approx(0.f).epsilon(eps));
    // Derivative: 1.0 for positive, 0.01 for negative
    CHECK(lay.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0, 1.0, 1.0, 1.0});
    CHECK(lay.activationFunctionDerivative({-0.5f, -0.5f, -0.5f, -0.5f})[0] ==
          doctest::Approx(0.01f).epsilon(eps));
  };

  auto testPReLU = [&eps, &alpha](const Layer &lay) {
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0f, 1.0f, 1.0f, 1.0f});
    CHECK(lay.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f})[1] ==
          doctest::Approx(0.f).epsilon(eps));
    CHECK(lay.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}) ==
          cv::Vec4f{1.0, 1.0, 1.0, 1.0});
    CHECK(lay.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})[1] ==
          doctest::Approx(alpha).epsilon(eps));
  };

  auto testReLU = [&eps](const Layer &lay) {
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f})[2] ==
          doctest::Approx(1.0f).epsilon(eps));
    CHECK(lay.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f})[3] ==
          doctest::Approx(0.0f).epsilon(eps));
    CHECK(lay.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f})[1] ==
          doctest::Approx(1.0f).epsilon(eps));
    CHECK(lay.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})[0] ==
          doctest::Approx(0.0f).epsilon(eps));
  };

  auto testSigm = [&eps](const Layer &lay) {
    CHECK(lay.activationFunction({0.0f, 0.0f, 0.0f, 0.0f})[0] ==
          doctest::Approx(0.5f).epsilon(eps));
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f})[1] ==
          doctest::Approx(0.731059f).epsilon(eps));
    // Derivative expects already-activated values: s*(1-s)
    // sigmoidDerivative(0.5) = 0.5 * 0.5 = 0.25
    CHECK(lay.activationFunctionDerivative({0.5f, 0.5f, 0.5f, 0.5f})[2] ==
          doctest::Approx(0.25f).epsilon(eps));
    // sigmoidDerivative(0.731059) = 0.731059 * 0.268941 = 0.196612
    CHECK(lay.activationFunctionDerivative({0.731059f, 0.731059f, 0.731059f, 0.731059f})[3] ==
          doctest::Approx(0.196612f).epsilon(eps));
  };

  auto testTanh = [&eps](const Layer &lay) {
    CHECK(lay.activationFunction({0.0f, 0.0f, 0.0f, 0.0f})[0] ==
          doctest::Approx(0.5).epsilon(eps));
    CHECK(lay.activationFunction({1.0f, 1.0f, 1.0f, 1.0f})[1] ==
          doctest::Approx(0.880797).epsilon(eps));
    CHECK(lay.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f})[2] ==
          doctest::Approx(0.119203).epsilon(eps));
    // Derivative expects already-activated values (in [0,1] range)
    // tanhDerivative(0.5) -> t=0, (1-0)/2 = 0.5
    cv::Vec4f input{0.5f, 0.5f, 0.5f, 0.5f};
    cv::Vec4f expected{0.5f, 0.5f, 0.5f, 0.5f};
    const auto output = lay.activationFunctionDerivative(input);
    for (int i = 0; i < 4; ++i) {
      CHECK(output[i] == doctest::Approx(expected[i]).epsilon(eps));
    }
    // tanhDerivative(0.880797) -> t=0.761594, (1-0.580026)/2 = 0.209987
    CHECK(lay.activationFunctionDerivative({0.880797f, 0.880797f, 0.880797f, 0.880797f})[3] ==
          doctest::Approx(0.209987f).epsilon(1e-4f));
    // tanhDerivative(0.119203) -> t=-0.761594, same result = 0.209987
    CHECK(lay.activationFunctionDerivative({0.119203f, 0.119203f, 0.119203f, 0.119203f})[0] ==
          doctest::Approx(0.209987f).epsilon(1e-4f));
  };

  auto testActivationFunction =
      [&hasAF, testELU, testLReLU, testPReLU, testReLU, testSigm,
       testTanh](const Layer &lay, const EActivationFunction &activ) {
        switch (activ) {
        case EActivationFunction::ELU:
          testELU(lay);
          hasAF.hasElu = true;
          break;
        case EActivationFunction::LReLU:
          testLReLU(lay);
          hasAF.hasLRelu = true;
          break;
        case EActivationFunction::PReLU:
          testPReLU(lay);
          hasAF.hasPRelu = true;
          break;
        case EActivationFunction::ReLU:
          testReLU(lay);
          hasAF.hasRelu = true;
          break;
        case EActivationFunction::Sigmoid:
          testSigm(lay);
          hasAF.hasSigm = true;
          break;
        case EActivationFunction::Tanh:
          testTanh(lay);
          hasAF.hasTanh = true;
          break;
        default:
          break;
        }
      };

  for (auto activ : {EActivationFunction::ELU, EActivationFunction::LReLU,
                     EActivationFunction::PReLU, EActivationFunction::ReLU,
                     EActivationFunction::Sigmoid, EActivationFunction::Tanh}) {
    networkParams = {
        .hidden_activation_alpha = alpha,
        .hidden_activation_function = activ,
    };
    builder.with(networkParams);
    CHECK_NOTHROW(builder.setActivationFunction());
    testActivationFunction(*hlayer, activ);
  }
  CHECK((hasAF.hasElu && hasAF.hasLRelu && hasAF.hasPRelu && hasAF.hasRelu &&
         hasAF.hasSigm && hasAF.hasTanh) == true);

  auto invalidEnum = static_cast<EActivationFunction>(900);
  networkParams = {.hidden_activation_function = invalidEnum};
  builder.with(networkParams);
  CHECK_THROWS_AS(builder.setActivationFunction(), NeuralNetworkException);

  // cleaning
  networkParams = {};
  builder.with(networkParams);
}