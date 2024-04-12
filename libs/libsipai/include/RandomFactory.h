/**
 * @file RandomFactory.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Random Factory
 * @date 2024-04-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include <memory>
#include <mutex>
#include <random>

namespace sipai {
class RandomFactory {
public:
  static RandomFactory &getInstance() {
    static std::once_flag initInstanceFlag;
    std::call_once(initInstanceFlag,
                   [] { instance_.reset(new RandomFactory); });
    return *instance_;
  }
  RandomFactory(RandomFactory const &) = delete;
  void operator=(RandomFactory const &) = delete;
  ~RandomFactory() = default;

  /**
   * @brief Generates a random number from a normal distribution.
   *
   * @param mean The mean of the normal distribution.
   * @param stddev The standard deviation of the normal distribution.
   * @return A random float number from the specified normal distribution.
   */
  float getRandom(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(gen);
  }

  /**
   * @brief Generates a random number from a normal distribution.
   * This function is a static shortcut to the getRandom member function.
   *
   * @param mean The mean of the normal distribution.
   * @param stddev The standard deviation of the normal distribution.
   * @return A random float number from the specified normal distribution.
   */
  static float Rand(float mean, float stddev) {
    return getInstance().getRandom(mean, stddev);
  }

private:
  RandomFactory() : gen(rd()) {}
  static std::unique_ptr<RandomFactory> instance_;

  std::random_device rd;
  std::mt19937 gen;
};
} // namespace sipai