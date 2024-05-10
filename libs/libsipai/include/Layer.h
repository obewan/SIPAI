/**
 * @file Layer.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Abstract layer class
 * @date 2023-08-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2023
 *
 */
#pragma once
#include "ActivationFunctions.h"
#include "Neuron.h"
#include "exception/NeuralNetworkException.h"
#include <atomic>
#include <cstddef>
#include <execution>
#include <functional>
#include <map>
#include <opencv2/opencv.hpp>
#include <thread>

namespace sipai {
class VulkanController;

using NeuronMat = std::vector<std::vector<Neuron>>;

enum class LayerType { LayerInput, LayerHidden, LayerOutput };

const std::map<std::string, LayerType, std::less<>> layer_map{
    {"LayerInput", LayerType::LayerInput},
    {"LayerHidden", LayerType::LayerHidden},
    {"LayerOutput", LayerType::LayerOutput}};

/**
 * @brief The Layer class represents a layer in a neural network. It contains a
 * vector of Neurons and has methods for forward propagation, backward
 * propagation, and updating weights.
 */
class Layer {
public:
  explicit Layer(LayerType layerType, size_t size_x = 0, size_t size_y = 0)
      : layerType(layerType), size_x(size_x), size_y(size_y) {
    neurons = NeuronMat(size_y, std::vector<Neuron>(size_x));
    for (size_t row = 0; row < size_y; ++row) {
      for (size_t col = 0; col < size_x; ++col) {
        neurons[row][col].index_x = col;
        neurons[row][col].index_y = row;
      }
    }
    values = cv::Mat((int)size_x, (int)size_y, CV_32FC4);
    errors = cv::Mat((int)size_x, (int)size_y, CV_32FC4);
  }
  virtual ~Layer() = default;

  const LayerType layerType;

  /**
   * @brief 2D vector of neurons in format [rows][cols], i.e. [y][x]
   *
   */
  NeuronMat neurons;

  /**
   * @brief 2D matrix of values, in format (x,y)
   *
   */
  cv::Mat values;

  /**
   * @brief 2D matrix of errors, in format (x,y)
   *
   */
  cv::Mat errors;

  /**
   * @brief previous layer, or nullptr if not exists
   *
   */
  Layer *previousLayer = nullptr;

  /**
   * @brief next layer, or nullptr if not exists
   *
   */
  Layer *nextLayer = nullptr;

  /**
   * @brief width (columns)
   *
   */
  size_t size_x = 0;

  /**
   * @brief height (rows)
   *
   */
  size_t size_y = 0;

  /**
   * @brief Get the layer total size, which is (size_x * size_y)
   *
   * @return size_t
   */
  size_t total() const { return size_x * size_y; }

  /**
   * @brief Get the indexes (x,y) of a neuron from the layer size() index
   *
   * @param index
   * @return std::pair<size_t, size_t>
   */
  std::pair<size_t, size_t> getPos(const size_t &index) const {
    if (index >= total()) {
      throw std::out_of_range("Index out of range");
    }
    size_t row = index / size_x;
    size_t col = index % size_x;
    return {row, col};
  }

  /**
   * @brief Get a neuron from a layer size() index
   *
   * @param index
   * @return Neuron&
   */
  Neuron &getNeuron(const size_t &index) {
    const auto &[row, col] = getPos(index);
    return neurons[row][col];
  }

  EActivationFunction eactivationFunction = EActivationFunction::ReLU;
  float activationFunctionAlpha = 0.0f;
  std::function<cv::Vec4f(cv::Vec4f)> activationFunction;
  std::function<cv::Vec4f(cv::Vec4f)> activationFunctionDerivative;

  const std::string UndefinedLayer = "UndefinedLayer";

  /**
   * @brief Apply a function on all the layer neurons
   *
   * @tparam Function a lambda function
   * @param operation
   */
  template <typename Function> void apply(Function operation) {
    for (auto &neuronRow : neurons) {
      for (auto &neuron : neuronRow) {
        operation(neuron);
      }
    }
  }

  /**
   * @brief Performs forward propagation using the previous layer.
   */
  virtual void forwardPropagation();

  /**
   * @brief Performs backward propagation using the next layer.
   */
  virtual void backwardPropagation(const float &error_min,
                                   const float &error_max);
  /**
   * @brief Updates the weights of the neurons in this layer using the
   * previous layer and a learning rate.
   *
   * @param learningRate The learning rate to use when updating weights.
   */
  virtual void updateWeights(float learningRate);

  const std::string getLayerTypeStr() const {
    for (const auto &[key, mLayerType] : layer_map) {
      if (mLayerType == layerType) {
        return key;
      }
    }
    return UndefinedLayer;
  }

  void
  setActivationFunction(const std::function<cv::Vec4f(cv::Vec4f)> &function,
                        const std::function<cv::Vec4f(cv::Vec4f)> &derivative) {
    activationFunction = function;
    activationFunctionDerivative = derivative;
  }
};
} // namespace sipai