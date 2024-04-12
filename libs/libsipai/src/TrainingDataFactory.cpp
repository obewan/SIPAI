#include "TrainingDataFactory.h"
#include "Manager.h"

using namespace sipai;

std::unique_ptr<TrainingDataFactory> TrainingDataFactory::instance_ = nullptr;

std::unique_ptr<ImagePartsPair> TrainingDataFactory::nextTraining() {
  return nullptr;
}

std::unique_ptr<ImagePartsPair> TrainingDataFactory::nextValidation() {
  return nullptr;
}

size_t TrainingDataFactory::trainingSize() { return 0; }
size_t TrainingDataFactory::validationSize() { return 0; }

void TrainingDataFactory::loadData() {}

void TrainingDataFactory::reset() {}