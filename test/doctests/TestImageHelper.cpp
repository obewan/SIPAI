#include "ImageHelper.h"
#include "doctest.h"
#include <filesystem>

using namespace sipai;

TEST_CASE("Testing ImageHelper") {

  SUBCASE("Test loadImage") {
    ImageHelper imageHelper;
    auto image = imageHelper.loadImage("../data/images/001a.png");
    CHECK(image.size().height > 0);
    CHECK(image.size().width > 0);
  }

  SUBCASE("Test saveImage") {
    ImageHelper imageHelper;
    auto image = imageHelper.loadImage("../data/images/001a.png");
    std::string tmpImage = "tmpImage.png";
    if (std::filesystem::exists(tmpImage)) {
      std::filesystem::remove(tmpImage);
    }
    CHECK(std::filesystem::exists(tmpImage) == false);
    imageHelper.saveImage(tmpImage, image);
    CHECK(std::filesystem::exists(tmpImage) == true);
    auto image2 = imageHelper.loadImage(tmpImage);
    CHECK(image2.size() == image.size());
    std::filesystem::remove(tmpImage);
  }

  SUBCASE("Test convertToRGBAVector") {
    ImageHelper imageHelper;
    auto image = imageHelper.loadImage("../data/images/001a.png");
    auto imageRgba = imageHelper.convertToRGBAVector(image);
    CHECK(imageRgba.size() > 0);
    CHECK(imageRgba.size() == image.size().area());
    for (const auto &rgba : imageRgba) {
      CHECK(rgba.isOutOfRange() == false);
    }
  }

  SUBCASE("Testing computeLoss method") {
    ImageHelper imageHelper;
    std::vector<RGBA> outputImage(10);
    std::vector<RGBA> targetImage(10);

    RGBA pixel(0.1f, 0.2f, 0.3f, 0.4f);
    for (int i = 0; i < 10; ++i) {
      outputImage[i] = pixel;
      targetImage[i] = pixel;
    }

    float loss = imageHelper.computeLoss(outputImage, targetImage);
    CHECK(loss == doctest::Approx(0.0));
  }

  SUBCASE("Testing computeLoss method with different images") {
    ImageHelper imageHelper;
    std::vector<RGBA> outputImage(10);
    std::vector<RGBA> targetImage(10);

    RGBA pixel1(0.1f, 0.2f, 0.3f, 0.4f);
    RGBA pixel2(0.5f, 0.6f, 0.7f, 0.8f);
    for (int i = 0; i < 10; ++i) {
      outputImage[i] = pixel1;
      targetImage[i] = pixel2;
    }

    float loss = imageHelper.computeLoss(outputImage, targetImage);
    CHECK(loss == doctest::Approx(0.16));
  }
}