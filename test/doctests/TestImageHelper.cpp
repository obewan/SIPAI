#include "ImageHelper.h"
#include "doctest.h"
#include <filesystem>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

using namespace sipai;

TEST_CASE("Testing ImageHelper") {

  SUBCASE("Test loadImage") {
    ImageHelper imageHelper;
    size_t split = 3;
    const auto &image =
        imageHelper.loadImage("../data/images/input/001a.png", split, false);
    for (const auto &part : image) {
      // Testing resize
      CHECK(part->data.size().width == 50);
      CHECK(part->data.size().height == 50);
      CHECK(part->data.total() == 50 * 50);
      // Testing converted to 4 channels
      int channels = part->data.channels();
      CHECK(channels == 4);
      // Testing that all values are in range [0,1]
      bool allValuesInRange = true;
      for (int x = 0; x < part->data.rows; ++x) {
        for (int y = 0; y < part->data.cols; ++y) {
          const cv::Vec4f &pixel = part->data.at<cv::Vec4f>(x, y);
          if (pixel[0] < 0.0f || pixel[0] > 1.0f || pixel[1] < 0.0f ||
              pixel[1] > 1.0f || pixel[2] < 0.0f || pixel[2] > 1.0f ||
              pixel[3] < 0.0f || pixel[3] > 1.0f) {
            allValuesInRange = false;
            break;
          }
        }
      }
      CHECK(allValuesInRange);
    }
  }

  SUBCASE("Test generateInputImage") {
    ImageHelper imageHelper;
    size_t split = 3;
    size_t reduce_factor = 4;
    const auto &imageTarget =
        imageHelper.loadImage("../data/images/target/001b.png", split, false);
    const auto &imageInput =
        imageHelper.generateInputImage(imageTarget, reduce_factor);
    for (const auto &part : imageInput) {
      // Testing resize (original size is 640x640)
      CHECK(part->data.size().width == 53);
      CHECK(part->data.size().height == 53);
      CHECK(part->data.total() == 53 * 53);
      // Testing converted to 4 channels
      int channels = part->data.channels();
      CHECK(channels == 4);
      // Testing that all values are in range [0,1]
      bool allValuesInRange = true;
      for (int x = 0; x < part->data.rows; ++x) {
        for (int y = 0; y < part->data.cols; ++y) {
          const cv::Vec4f &pixel = part->data.at<cv::Vec4f>(x, y);
          if (pixel[0] < 0.0f || pixel[0] > 1.0f || pixel[1] < 0.0f ||
              pixel[1] > 1.0f || pixel[2] < 0.0f || pixel[2] > 1.0f ||
              pixel[3] < 0.0f || pixel[3] > 1.0f) {
            allValuesInRange = false;
            break;
          }
        }
      }
      CHECK(allValuesInRange);
    }
  }

  SUBCASE("Test saveImage") {
    ImageHelper imageHelper;
    size_t split = 2;
    const auto &image =
        imageHelper.loadImage("../data/images/input/001a.png", split, true);
    std::string tmpImage = "tmpImage.png";
    if (std::filesystem::exists(tmpImage)) {
      std::filesystem::remove(tmpImage);
    }
    CHECK_FALSE(std::filesystem::exists(tmpImage));
    imageHelper.saveImage(tmpImage, image, split);
    CHECK(std::filesystem::exists(tmpImage));
    const auto &image2 = imageHelper.loadImage(tmpImage, split, true);
    for (size_t i = 0; i < image2.size(); i++) {
      CHECK(image2[i]->data.size().width == image[i]->data.size().width);
      CHECK(image2[i]->data.size().height == image[i]->data.size().height);
    }
    std::filesystem::remove(tmpImage);
  }

  SUBCASE("Testing computeLoss method") {
    ImageHelper imageHelper;
    cv::Mat outputData = cv::Mat::ones(3, 3, CV_32FC1) * 5.0f;
    cv::Mat targetData = cv::Mat::ones(3, 3, CV_32FC1) * 5.0f;

    float loss = imageHelper.computeLoss(outputData, targetData);
    CHECK(loss == doctest::Approx(0.0));
  }

  SUBCASE("Testing computeLoss method with different images") {
    ImageHelper imageHelper;
    cv::Mat outputData(10, 1, CV_32FC4, cv::Vec4f(0.1f, 0.2f, 0.3f, 0.4f));
    cv::Mat targetData(10, 1, CV_32FC4, cv::Vec4f(0.5f, 0.6f, 0.7f, 0.8f));

    float loss = imageHelper.computeLoss(outputData, targetData);
    CHECK(loss == doctest::Approx(0.16));
  }
}