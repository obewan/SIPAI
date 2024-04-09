#include "ImageHelper.h"
#include "doctest.h"
#include <filesystem>

using namespace sipai;

TEST_CASE("Testing ImageHelper") {

  SUBCASE("Test loadImage") {
    ImageHelper imageHelper;
    const auto &image = imageHelper.loadImage("../data/images/001a.png");
    CHECK(image.size_x * image.size_y > 0);
    for (const auto &rgba : image.data) {
      CHECK_FALSE(rgba.isOutOfRange());
    }
  }

  SUBCASE("Test saveImage") {
    ImageHelper imageHelper;
    const auto &image = imageHelper.loadImage("../data/images/001a.png");
    std::string tmpImage = "tmpImage.png";
    if (std::filesystem::exists(tmpImage)) {
      std::filesystem::remove(tmpImage);
    }
    CHECK_FALSE(std::filesystem::exists(tmpImage));
    imageHelper.saveImage(tmpImage, image);
    CHECK(std::filesystem::exists(tmpImage));
    const auto &image2 = imageHelper.loadImage(tmpImage);
    CHECK(image2.size_x == image.size_x);
    CHECK(image2.size_y == image.size_y);
    std::filesystem::remove(tmpImage);
  }

  SUBCASE("Testing computeLoss method") {
    ImageHelper imageHelper;
    std::vector<RGBA> outputData(10);
    std::vector<RGBA> targetData(10);

    RGBA pixel(0.1f, 0.2f, 0.3f, 0.4f);
    for (int i = 0; i < 10; ++i) {
      outputData[i] = pixel;
      targetData[i] = pixel;
    }

    float loss = imageHelper.computeLoss(outputData, targetData);
    CHECK(loss == doctest::Approx(0.0));
  }

  SUBCASE("Testing computeLoss method with different images") {
    ImageHelper imageHelper;
    std::vector<RGBA> outputData(10);
    std::vector<RGBA> targetData(10);

    RGBA pixel1(0.1f, 0.2f, 0.3f, 0.4f);
    RGBA pixel2(0.5f, 0.6f, 0.7f, 0.8f);
    for (int i = 0; i < 10; ++i) {
      outputData[i] = pixel1;
      targetData[i] = pixel2;
    }

    float loss = imageHelper.computeLoss(outputData, targetData);
    CHECK(loss == doctest::Approx(0.16));
  }
}