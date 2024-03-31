#include "ActivationFunctions.h"
#include "LayerHidden.h"
#include "NeuralNetwork.h"
#include "doctest.h"
#include "exception/NetworkException.h"

using namespace sipai;

TEST_CASE("Testing the Activation Functions") {

  auto network = new NeuralNetwork();
  auto hlayer = new LayerHidden();
  Neuron n1;
  Neuron n2;
  hlayer->neurons.push_back(n1);
  hlayer->neurons.push_back(n2);
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

  auto testELU = [&eps](const Neuron &neu) {
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(0) ==
          doctest::Approx(0).epsilon(eps));
    CHECK(neu.activationFunction({0.0f, 0.0f, 0.0f, 0.0f}).value ==
          RGBA{0.0f, 0.0f, 0.0f, 0.0f}.value);
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(0) == doctest::Approx(0.0367879f).epsilon(eps));
  };

  auto testLReLU = [&eps](const Neuron &neu) {
    RGBA input{1.0f, 1.0f, 1.0f, 1.0f};
    RGBA expected{0.01f, 0.01f, 0.01f, 0.01f};
    const auto output = neu.activationFunction(input).value;
    for (size_t i = 0; i < 4; ++i) {
      CHECK(output[i] == doctest::Approx(expected.value[i]).epsilon(eps));
    }
    CHECK(neu.activationFunction({-0.5f, -0.5f, -0.5f, -0.5f}).value.at(0) ==
          doctest::Approx(0.f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunctionDerivative({-0.5f, -0.5f, -0.5f, -0.5f})
              .value.at(0) == doctest::Approx(0.01f).epsilon(eps));
  };

  auto testPReLU = [&eps, &alpha](const Neuron &neu) {
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0f, 1.0f, 1.0f, 1.0f}.value);
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(1) ==
          doctest::Approx(0.f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(1) == doctest::Approx(alpha).epsilon(eps));
  };

  auto testReLU = [&eps](const Neuron &neu) {
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value.at(2) ==
          doctest::Approx(1.0f).epsilon(eps));
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(3) ==
          doctest::Approx(0.0f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f})
              .value.at(1) == doctest::Approx(1.0f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(0) == doctest::Approx(0.0f).epsilon(eps));
  };

  auto testSigm = [&eps](const Neuron &neu) {
    CHECK(neu.activationFunction({0.0f, 0.0f, 0.0f, 0.0f}).value.at(0) ==
          doctest::Approx(0.5f).epsilon(eps));
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value.at(1) ==
          doctest::Approx(0.731059f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({0.0f, 0.0f, 0.0f, 0.0f})
              .value.at(2) == doctest::Approx(0.25f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f})
              .value.at(3) == doctest::Approx(0.196612f).epsilon(eps));
  };

  auto testTanh = [&eps](const Neuron &neu) {
    CHECK(neu.activationFunction({0.0f, 0.0f, 0.0f, 0.0f}).value.at(0) ==
          doctest::Approx(0.5).epsilon(eps));
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value.at(1) ==
          doctest::Approx(0.880797).epsilon(eps));
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(2) ==
          doctest::Approx(0.119203).epsilon(eps));
    RGBA input{0.0f, 0.0f, 0.0f, 0.0f};
    RGBA expected{0.75f, 0.75f, 0.75f, 0.75f};
    const auto output = neu.activationFunctionDerivative(input).value;
    for (size_t i = 0; i < 4; ++i) {
      CHECK(output[i] == doctest::Approx(expected.value[i]).epsilon(eps));
    }
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f})
              .value.at(3) == doctest::Approx(0.224196).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(0) == doctest::Approx(0.985791).epsilon(eps));
  };

  auto testActivationFunction =
      [&hasAF, testELU, testLReLU, testPReLU, testReLU, testSigm,
       testTanh](const Neuron &neu, const EActivationFunction &activ) {
        switch (activ) {
        case EActivationFunction::ELU:
          testELU(neu);
          hasAF.hasElu = true;
          break;
        case EActivationFunction::LReLU:
          testLReLU(neu);
          hasAF.hasLRelu = true;
          break;
        case EActivationFunction::PReLU:
          testPReLU(neu);
          hasAF.hasPRelu = true;
          break;
        case EActivationFunction::ReLU:
          testReLU(neu);
          hasAF.hasRelu = true;
          break;
        case EActivationFunction::Sigmoid:
          testSigm(neu);
          hasAF.hasSigm = true;
          break;
        case EActivationFunction::Tanh:
          testTanh(neu);
          hasAF.hasTanh = true;
          break;
        default:
          break;
        }
      };

  for (auto activ : {EActivationFunction::ELU, EActivationFunction::LReLU,
                     EActivationFunction::PReLU, EActivationFunction::ReLU,
                     EActivationFunction::Sigmoid, EActivationFunction::Tanh}) {
    CHECK_NOTHROW(network->SetActivationFunction(hlayer, activ, alpha));
    for (const auto &neu : hlayer->neurons) {
      testActivationFunction(neu, activ);
    }
  }
  CHECK((hasAF.hasElu && hasAF.hasLRelu && hasAF.hasPRelu && hasAF.hasRelu &&
         hasAF.hasSigm && hasAF.hasTanh) == true);

  auto invalidEnum = static_cast<EActivationFunction>(900);
  CHECK_THROWS_AS(network->SetActivationFunction(hlayer, invalidEnum, 0.1f),
                  NetworkException);

  CHECK_NOTHROW(delete hlayer);
  CHECK_NOTHROW(delete network);
}