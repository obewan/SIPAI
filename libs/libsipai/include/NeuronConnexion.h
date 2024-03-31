/**
 * @file NeuronConnexion.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Connexion between neurons
 * @date 2024-03-10
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include "RGBA.h"

namespace sipai {
class Neuron;
class NeuronConnection {
public:
  Neuron *neuron;
  RGBA weight;

  NeuronConnection(Neuron *neuron, RGBA weight)
      : neuron(neuron), weight(weight) {}
};
} // namespace sipai