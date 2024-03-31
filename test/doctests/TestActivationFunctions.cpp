#include "ActivationFunctions.h"
#include "HiddenLayer.h"
#include "NeuralNetwork.h"
#include "doctest.h"
#include "exception/NetworkException.h"

using namespace sipai;

TEST_CASE("Testing the Activation Functions") {

  auto network = new NeuralNetwork();
  auto hlayer = new HiddenLayer();
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
          doctest::Approx(-0.0632121).epsilon(eps));
    CHECK(neu.activationFunction({0.0f, 0.0f, 0.0f, 0.0f}).value ==
          RGBA{0.0f, 0.0f, 0.0f, 0.0f}.value);
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(0) == doctest::Approx(0.0367879f).epsilon(eps));
  };

  auto testLReLU = [&eps](const Neuron &neu) {
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunction({-0.5f, -0.5f, -0.5f, -0.5f}).value.at(0) ==
          doctest::Approx(-0.005f).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0, 1.0, 1.0, 1.0}.value);
    CHECK(neu.activationFunctionDerivative({-0.5f, -0.5f, -0.5f, -0.5f})
              .value.at(0) == doctest::Approx(0.01f).epsilon(eps));
  };

  auto testPReLU = [&eps, &alpha](const Neuron &neu) {
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value ==
          RGBA{1.0f, 1.0f, 1.0f, 1.0f}.value);
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(1) ==
          doctest::Approx(-0.1f).epsilon(eps));
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
          doctest::Approx(0.0).epsilon(eps));
    CHECK(neu.activationFunction({1.0f, 1.0f, 1.0f, 1.0f}).value.at(1) ==
          doctest::Approx(0.7615941559557649).epsilon(eps));
    CHECK(neu.activationFunction({-1.0f, -1.0f, -1.0f, -1.0f}).value.at(2) ==
          doctest::Approx(-0.7615941559557649).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({0.0f, 0.0f, 0.0f, 0.0f}).value ==
          RGBA{1.f, 1.f, 1.f, 1.f}.value);
    CHECK(neu.activationFunctionDerivative({1.0f, 1.0f, 1.0f, 1.0f})
              .value.at(3) ==
          doctest::Approx(0.41997434161402614).epsilon(eps));
    CHECK(neu.activationFunctionDerivative({-1.0f, -1.0f, -1.0f, -1.0f})
              .value.at(0) == doctest::Approx(0.419974f).epsilon(eps));
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