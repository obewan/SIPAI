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
  }
}