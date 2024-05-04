#include "Common.h"
#include "doctest.h"
#include <opencv2/core/matx.hpp>
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

  SUBCASE("Test clamps") {
    // Test cv::Vec4f clamp
    cv::Vec4f vec4{0.4f, -2.2f, 2.3f, 0.5f};
    cv::Vec4f clamped = clamp4f(vec4, cv::Vec4f(0.0f, 0.0f, 0.0f, 0.0f),
                                cv::Vec4f(1.0f, 1.0f, 1.0f, 1.0f));
    CHECK(clamped == cv::Vec4f{0.4f, 0.0f, 1.0f, 0.5f});
    cv::Vec4f clamped2 = clamp4f(vec4, cv::Vec4f(0.0f, -1.0f, 0.0f, 0.0f),
                                 cv::Vec4f(1.0f, 1.0f, 2.0f, 1.0f));
    CHECK(clamped2 == cv::Vec4f{0.4f, -1.0f, 2.0f, 0.5f});
    cv::Vec4f clamped3 = clamp4f(vec4);
    CHECK(clamped3 == cv::Vec4f{0.4f, 0.0f, 1.0f, 0.5f});

    // Test cv::Mat clamp
    cv::Mat inputMat(2, 2, CV_32FC4);
    inputMat.setTo(cv::Scalar(0.5f, 2.8f, 1.2f, -3.9f));
    cv::Mat clampedMat = mat_clamp4f(inputMat);
    for (int x = 0; x < clampedMat.rows; ++x) {
      for (int y = 0; y < clampedMat.cols; ++y) {
        const cv::Vec4f &pixel = clampedMat.at<cv::Vec4f>(x, y);
        CHECK(pixel[0] >= 0.0);
        CHECK(pixel[0] <= 1.0);
        CHECK(pixel[1] >= 0.0);
        CHECK(pixel[1] <= 1.0);
        CHECK(pixel[2] >= 0.0);
        CHECK(pixel[2] <= 1.0);
        CHECK(pixel[3] >= 0.0);
        CHECK(pixel[3] <= 1.0);
      }
    }
    cv::Mat clampedMat2 =
        mat_clamp4f(inputMat, cv::Vec4f(0.0f, 0.0f, 0.0f, -1.0f),
                    cv::Vec4f(1.0f, 2.0f, 1.0f, 0.0f));
    for (int x = 0; x < clampedMat2.rows; ++x) {
      for (int y = 0; y < clampedMat2.cols; ++y) {
        const cv::Vec4f &pixel = clampedMat2.at<cv::Vec4f>(x, y);
        CHECK(pixel[0] >= 0.0);
        CHECK(pixel[0] <= 1.0);
        CHECK(pixel[1] >= 0.0);
        CHECK(pixel[1] <= 2.0);
        CHECK(pixel[2] >= 0.0);
        CHECK(pixel[2] <= 1.0);
        CHECK(pixel[3] >= -1.0);
        CHECK(pixel[3] <= 0.0);
      }
    }
  }
}
