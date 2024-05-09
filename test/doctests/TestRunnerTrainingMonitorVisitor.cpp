#include "Manager.h"
#include "NeuralNetwork.h"
#include "RunnerTrainingMonitoredVisitor.h"
#include "TrainingDataFactory.h"
#include "doctest.h"
#include "exception/RunnerVisitorException.h"
#include "exception/TrainingDataFactoryException.h"
#include <filesystem>
#include <memory>

using namespace sipai;

TEST_CASE("Testing RunnerTrainingMonitoredVisitor") {
  SUBCASE("Test exceptions") {
    RunnerTrainingMonitoredVisitor visitor;
    TrainingDataFactory::getInstance().clear();
    auto &manager = Manager::getInstance();

    // no network
    manager.network.reset();
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    // no data
    manager.network = std::make_unique<NeuralNetwork>();
    manager.app_params.training_data_file = "";
    manager.app_params.training_data_folder = "";
    CHECK_THROWS_AS(visitor.visit(), RunnerVisitorException);

    TrainingDataFactory::getInstance().clear();
  }
}