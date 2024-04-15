#include "Common.h"
#include "doctest.h"
#include <thread>

using namespace sipai;

TEST_CASE("Testing Common") {

  SUBCASE("Test getHMS") {
    using namespace std::chrono_literals;

    const auto start{std::chrono::steady_clock::now()};
    auto test1 = getHMSfromS(3662);
    auto expect1 = std::array<size_t, 3>{1, 1, 2};
    for (int i = 0; i < 3; i++) {
      CHECK(test1[i] == expect1[i]);
    }

    std::this_thread::sleep_for(1000ms);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration elapsed_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto test2 = getHMSfromS(elapsed_seconds.count());
    CHECK(test2[0] == 0);
    CHECK(test2[1] == 0);
    CHECK(test2[2] > 0);
    CHECK(test2[2] < 5);
  }
}
