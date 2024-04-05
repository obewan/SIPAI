#include "RGBA.h"
#include "doctest.h"

using namespace sipai;

TEST_CASE("Testing the RGBA struct") {
  SUBCASE("Test random") {
    auto rgba1 = RGBA().random(4.0f);
    auto rgba2 = RGBA().random(4.0f);

    // beware there's a small chance that the values are equals,
    // very small chance though (random)
    CHECK_FALSE(std::equal(rgba1.value.begin(), rgba1.value.end(),
                           rgba2.value.begin()));
    CHECK_FALSE(rgba1.isOutOfRange());
    CHECK_FALSE(rgba2.isOutOfRange());
  }

  SUBCASE("Test toStringCsv") {
    auto rgba = RGBA(0.1, 0.2, 0.3, 0.4);
    CHECK(rgba.toStringCsv() == "0.100,0.200,0.300,0.400");
  }
}
