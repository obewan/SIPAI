/**
 * @file Neuron.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Neuron class
 * @date 2024-03-07
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "ActivationFunctions.h"
#include "NeuronConnection.h"
#include <exception>
#include <functional>
#include <math.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <vector>

namespace sipai {

/**
 * @brief The Neuron class represents a neuron in a neural network. It contains
 * a value, bias, error, and a vector of weights. It also has methods for
 * initializing weights.
 */
class Neuron {
public:
  // Default constructor
  Neuron() = default;

  // The weights of the neuron
  cv::Mat weights;

  // Index in current layer
  size_t index_x;
  size_t index_y;

  // Some indexes to use with Vulkan
  mutable size_t weightsIndex = 0;
  mutable size_t neighborsIndex = 0;
  mutable size_t neighborsSize = 0;

  // Connections to the adjacents neurons in the same layer, using
  // 4-neighborhood (Von Neumann neighborhood). Could be improve to
  // 8-neighborhood (Moore neighborhood) or Extended neighborhood (radius)
  // later.
  std::vector<NeuronConnection> neighbors;

  /**
   * @brief Initializes the weights of the neuron to a given size. The weights
   * are randomized to break symmetry.
   *
   * @param size_x The new size in X of the weights vector.
   * @param size_y The new size in Y of the weights vector.
   */
  void initWeights(size_t size_x, size_t size_y) {
    weights = cv::Mat((int)size_x, (int)size_y, CV_32FC4);

    // Random initialization
    cv::randn(weights, cv::Vec4f::all(0), cv::Vec4f::all(1));
  }

  std::string toStringCsv(size_t max_weights) const {
    std::ostringstream oss;
    for (int i = 0; i < weights.rows; i++) {
      for (int j = 0; j < weights.cols; j++) {
        for (int k = 0; k < 4; k++) {
          oss << weights.at<cv::Vec4f>(j, i)[k] << ",";
        }
      }
    };

    // fill the lasts columns with empty ",RGBA"
    for (size_t i = weights.total(); i < max_weights; ++i) {
      oss << ",,,,";
    }
    std::string str = oss.str();
    if (!str.empty()) {
      str.pop_back(); // remove the extra comma
    }
    return str;
  }

  std::string toNeighborsStringCsv(size_t max_weights) const {
    std::ostringstream oss;
    for (const auto &neighbor : neighbors) {
      for (int i = 0; i < 4; i++) {
        oss << neighbor.weight[i] << ",";
      }
    }
    // fill the lasts columns with empty ",RGBA"
    for (size_t i = neighbors.size(); i < max_weights; ++i) {
      oss << ",,,,";
    }
    std::string str = oss.str();
    if (!str.empty()) {
      str.pop_back(); // remove the extra comma
    }
    return str;
  }
};
} // namespace sipai