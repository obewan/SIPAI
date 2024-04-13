#include "ImageHelper.h"
#include "doctest.h"
#include <filesystem>

using namespace sipai;

TEST_CASE("Testing ImageHelper") {

  SUBCASE("Test loadImage") {
    ImageHelper imageHelper;
    size_t split = 3;
    const auto &image =
        imageHelper.loadImage("../data/images/001a.png", split, false);
    for (const auto &part : image) {
      CHECK(part->size_x == 50);
      CHECK(part->size_y == 50);
      CHECK(part->size() == 50 * 50);
      CHECK(part->data.size() == part->size());
      for (const auto &rgba : part->data) {
        CHECK_FALSE(rgba.isOutOfRange());
      }
    }
  }

  SUBCASE("Test saveImage") {
    ImageHelper imageHelper;
    size_t split = 2;
    const auto &image =
        imageHelper.loadImage("../data/images/001a.png", split, true);
    std::string tmpImage = "tmpImage.png";
    if (std::filesystem::exists(tmpImage)) {
      std::filesystem::remove(tmpImage);
    }
    CHECK_FALSE(std::filesystem::exists(tmpImage));
    imageHelper.saveImage(tmpImage, image, split);
    CHECK(std::filesystem::exists(tmpImage));
    const auto &image2 = imageHelper.loadImage(tmpImage, split, true);
    for (size_t i = 0; i < image2.size(); i++) {
      CHECK(image2[i]->size_x == image[i]->size_x);
      CHECK(image2[i]->size_y == image[i]->size_y);
    }
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