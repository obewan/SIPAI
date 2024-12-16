/**
 * @file NeuronConnection.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Connection between neurons
 * @date 2024-03-10
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include <opencv2/opencv.hpp>

namespace sipai {
class Neuron;
class NeuronConnection {
public:
  Neuron *neuron;
  cv::Vec4f weight;

  NeuronConnection(Neuron *neuron, cv::Vec4f weight)
      : neuron(neuron), weight(weight) {}
};
} // namespace sipai