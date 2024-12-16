/**
 * @file BindingNetworkParams.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Bindings for network_params
 * @date 2024-08-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include "ActivationFunctions.h"
#include "Common.h"
#include "NeuralNetworkParams.h"
#include <QObject>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class BindingNetworkParams : public QObject {
  Q_OBJECT
  Q_PROPERTY(
      int activationFunctionHidden READ getActivationFunctionHidden WRITE
          setActivationFunctionHidden NOTIFY activationFunctionHiddenChanged)
  Q_PROPERTY(
      int activationFunctionOutput READ getActivationFunctionOutput WRITE
          setActivationFunctionOutput NOTIFY activationFunctionOutputChanged)
  Q_PROPERTY(double activationAlphaHidden READ getActivationAlphaHidden WRITE
                 setActivationAlphaHidden NOTIFY activationAlphaHiddenChanged)
  Q_PROPERTY(double activationAlphaOutput READ getActivationAlphaOutput WRITE
                 setActivationAlphaOutput NOTIFY activationAlphaOutputChanged)
  Q_PROPERTY(int hiddenLayersCount READ getHiddenLayersCount WRITE
                 setHiddenLayersCount NOTIFY hiddenLayersCountChanged)
  Q_PROPERTY(int inputNeuronsX READ getInputNeuronsX WRITE setInputNeuronsX
                 NOTIFY inputNeuronsXChanged)
  Q_PROPERTY(int inputNeuronsY READ getInputNeuronsY WRITE setInputNeuronsY
                 NOTIFY inputNeuronsYChanged)
  Q_PROPERTY(int hiddenNeuronsX READ getHiddenNeuronsX WRITE setHiddenNeuronsX
                 NOTIFY hiddenNeuronsXChanged)
  Q_PROPERTY(int hiddenNeuronsY READ getHiddenNeuronsY WRITE setHiddenNeuronsY
                 NOTIFY hiddenNeuronsYChanged)
  Q_PROPERTY(int outputNeuronsX READ getOutputNeuronsX WRITE setOutputNeuronsX
                 NOTIFY outputNeuronsXChanged)
  Q_PROPERTY(int outputNeuronsY READ getOutputNeuronsY WRITE setOutputNeuronsY
                 NOTIFY outputNeuronsYChanged)
  Q_PROPERTY(double learningRate READ getLearningRate WRITE setLearningRate
                 NOTIFY learningRateChanged)
  Q_PROPERTY(bool adaptiveLearningRate READ getAdaptiveLearningRate WRITE
                 setAdaptiveLearningRate NOTIFY adaptiveLearningRateChanged)
  Q_PROPERTY(
      double adaptiveLearningRateFactor READ getAdaptiveLearningRateFactor WRITE
          setAdaptiveLearningRateFactor NOTIFY
              adaptiveLearningRateFactorChanged)
  Q_PROPERTY(
      bool adaptiveLearningRateIncrease READ getAdaptiveLearningRateIncrease
          WRITE setAdaptiveLearningRateIncrease NOTIFY
              adaptiveLearningRateIncreaseChanged)
  Q_PROPERTY(
      double errorMin READ getErrorMin WRITE setErrorMin NOTIFY errorMinChanged)
  Q_PROPERTY(
      double errorMax READ getErrorMax WRITE setErrorMax NOTIFY errorMaxChanged)

public:
  BindingNetworkParams(QObject *parent = nullptr);
  void connectUi(Ui::MainWindow *ui);
  void reload();

  int getActivationFunctionHidden() const {
    return static_cast<int>(network_params.hidden_activation_function);
  }
  void setActivationFunctionHidden(int value) {
    auto evalue = static_cast<sipai::EActivationFunction>(value);
    if (network_params.hidden_activation_function != evalue) {
      network_params.hidden_activation_function = evalue;
      emit activationFunctionHiddenChanged(value);
    }
  }

  int getActivationFunctionOutput() const {
    return static_cast<int>(network_params.output_activation_function);
  }
  void setActivationFunctionOutput(int value) {
    auto evalue = static_cast<sipai::EActivationFunction>(value);
    if (network_params.output_activation_function != evalue) {
      network_params.output_activation_function = evalue;
      emit activationFunctionOutputChanged(value);
    }
  }

  double getActivationAlphaHidden() const {
    return static_cast<double>(network_params.hidden_activation_alpha);
  }
  void setActivationAlphaHidden(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.hidden_activation_alpha != fvalue) {
      network_params.hidden_activation_alpha = fvalue;
      emit activationAlphaHiddenChanged(value);
    }
  }

  double getActivationAlphaOutput() const {
    return static_cast<double>(network_params.output_activation_alpha);
  }
  void setActivationAlphaOutput(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.output_activation_alpha != fvalue) {
      network_params.output_activation_alpha = fvalue;
      emit activationAlphaOutputChanged(value);
    }
  }

  int getInputLayersCount() const {
    return 1; // no multi input layers for now (and for a long time)
  }
  void setInputLayersCount(int value) {
    // stub
  }

  int getHiddenLayersCount() const {
    return static_cast<int>(network_params.hiddens_count);
  }
  void setHiddenLayersCount(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.hiddens_count != svalue) {
      network_params.hiddens_count = svalue;
      emit hiddenLayersCountChanged(value);
    }
  }

  int getOutputLayersCount() const {
    return 1; // no multi output layers for now (and for a long time)
  }
  void setOutputLayersCount(int value) {
    // stub
  }

  int getInputNeuronsX() const {
    return static_cast<int>(network_params.input_size_x);
  }
  void setInputNeuronsX(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.input_size_x != svalue) {
      network_params.input_size_x = svalue;
      emit inputNeuronsXChanged(value);
    }
  }

  int getInputNeuronsY() const {
    return static_cast<int>(network_params.input_size_y);
  }
  void setInputNeuronsY(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.input_size_y != svalue) {
      network_params.input_size_y = svalue;
      emit inputNeuronsYChanged(value);
    }
  }

  int getHiddenNeuronsX() const {
    return static_cast<int>(network_params.hidden_size_x);
  }
  void setHiddenNeuronsX(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.hidden_size_x != svalue) {
      network_params.hidden_size_x = svalue;
      emit hiddenNeuronsXChanged(value);
    }
  }

  int getHiddenNeuronsY() const {
    return static_cast<int>(network_params.hidden_size_y);
  }
  void setHiddenNeuronsY(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.hidden_size_y != svalue) {
      network_params.hidden_size_y = svalue;
      emit hiddenNeuronsYChanged(value);
    }
  }

  int getOutputNeuronsX() const {
    return static_cast<int>(network_params.output_size_x);
  }
  void setOutputNeuronsX(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.output_size_x != svalue) {
      network_params.output_size_x = svalue;
      emit outputNeuronsXChanged(value);
    }
  }

  int getOutputNeuronsY() const {
    return static_cast<int>(network_params.output_size_y);
  }
  void setOutputNeuronsY(int value) {
    auto svalue = static_cast<size_t>(value);
    if (network_params.output_size_y != svalue) {
      network_params.output_size_y = svalue;
      emit outputNeuronsYChanged(value);
    }
  }

  double getLearningRate() const {
    return static_cast<double>(network_params.learning_rate);
  }
  void setLearningRate(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.learning_rate != fvalue) {
      network_params.learning_rate = fvalue;
      emit learningRateChanged(value);
    }
  }

  bool getAdaptiveLearningRate() const {
    return network_params.adaptive_learning_rate;
  }
  void setAdaptiveLearningRate(bool value) {
    if (network_params.adaptive_learning_rate != value) {
      network_params.adaptive_learning_rate = value;
      emit adaptiveLearningRateChanged(value);
    }
  }

  double getAdaptiveLearningRateFactor() const {
    return static_cast<double>(network_params.adaptive_learning_rate_factor);
  }
  void setAdaptiveLearningRateFactor(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.adaptive_learning_rate_factor != fvalue) {
      network_params.adaptive_learning_rate_factor = fvalue;
      emit adaptiveLearningRateFactorChanged(value);
    }
  }

  bool getAdaptiveLearningRateIncrease() const {
    return network_params.enable_adaptive_increase;
  }
  void setAdaptiveLearningRateIncrease(bool value) {
    if (network_params.enable_adaptive_increase != value) {
      network_params.enable_adaptive_increase = value;
      emit adaptiveLearningRateIncreaseChanged(value);
    }
  }

  double getErrorMin() const {
    return static_cast<double>(network_params.error_min);
  }
  void setErrorMin(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.error_min != fvalue) {
      network_params.error_min = fvalue;
      emit errorMinChanged(value);
    }
  }

  double getErrorMax() const {
    return static_cast<double>(network_params.error_max);
  }
  void setErrorMax(double value) {
    auto fvalue = static_cast<float>(value);
    if (network_params.error_max != fvalue) {
      network_params.error_max = fvalue;
      emit errorMaxChanged(value);
    }
  }

signals:
  void activationFunctionHiddenChanged(int value);
  void activationFunctionOutputChanged(int value);
  void activationAlphaHiddenChanged(double value);
  void activationAlphaOutputChanged(double value);
  void hiddenLayersCountChanged(int value);
  void inputNeuronsXChanged(int value);
  void inputNeuronsYChanged(int value);
  void hiddenNeuronsXChanged(int value);
  void hiddenNeuronsYChanged(int value);
  void outputNeuronsXChanged(int value);
  void outputNeuronsYChanged(int value);
  void learningRateChanged(double value);
  void adaptiveLearningRateChanged(bool value);
  void adaptiveLearningRateFactorChanged(double value);
  void adaptiveLearningRateIncreaseChanged(bool value);
  void errorMinChanged(double value);
  void errorMaxChanged(double value);

private:
  sipai::NeuralNetworkParams &network_params;
};