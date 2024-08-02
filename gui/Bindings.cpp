#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "Bindings.h"
#include "MainWindow.h"

#include <algorithm>
#include <ranges>

using namespace Qt::StringLiterals;
using namespace sipai;

void Bindings::setBindings(Ui::MainWindow *ui) {
  // bind some params
  connect(
      ui->lineEditCurrentNetwork, &QLineEdit::textChanged, this,
      [this](const QString &text) { setNetworkToImport(text.toStdString()); });
  connect(ui->spinBoxInputNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setInputNeuronsX(value); });
  connect(ui->spinBoxInputNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setInputNeuronsY(value); });
  connect(ui->spinBoxHiddenNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setHiddenNeuronsX(value); });
  connect(ui->spinBoxHiddenNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setHiddenNeuronsY(value); });
  connect(ui->spinBoxOutputNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setOutputNeuronsX(value); });
  connect(ui->spinBoxOutputNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setOutputNeuronsY(value); });

  // add and bind activation functions
  for (const auto &[key, value] : activation_map) {
    ui->comboBoxActivationFunctionHidden->addItem(QString::fromStdString(key),
                                                  static_cast<int>(value));
    ui->comboBoxActivationFunctionOutput->addItem(QString::fromStdString(key),
                                                  static_cast<int>(value));
  }
  connect(ui->comboBoxActivationFunctionHidden, &QComboBox::currentIndexChanged,
          this, [this](int value) {
            setActivationFunctionHidden(
                static_cast<EActivationFunction>(value));
          });
  connect(ui->comboBoxActivationFunctionOutput, &QComboBox::currentIndexChanged,
          this, [this](int value) {
            setActivationFunctionOutput(
                static_cast<EActivationFunction>(value));
          });

  // add and bind running mode
  for (const auto &[key, value] : mode_map) {
    ui->comboBoxMode->addItem(QString::fromStdString(key),
                              static_cast<int>(value));
  }
}

void Bindings::getAppParams(Ui::MainWindow *ui) {
  const auto &app_params = Manager::getConstInstance().app_params;
  ui->lineEditCurrentNetwork->setText(app_params.network_to_import.c_str());
  ui->comboBoxMode->setCurrentIndex(static_cast<int>(app_params.run_mode));
}

void Bindings::getNetworkParams(Ui::MainWindow *ui) {
  const auto &network_params = Manager::getConstInstance().network_params;
  const auto &network = Manager::getConstInstance().network;

  int inputCount = 1;
  int hiddenCount = (int)network_params.hiddens_count;
  int outputCount = 1;
  if (network) {
    inputCount = std::ranges::count_if(network->layers, [](Layer *layer) {
      return layer->layerType == LayerType::LayerInput;
    });
    hiddenCount = std::ranges::count_if(network->layers, [](Layer *layer) {
      return layer->layerType == LayerType::LayerHidden;
    });
    outputCount = std::ranges::count_if(network->layers, [](Layer *layer) {
      return layer->layerType == LayerType::LayerOutput;
    });
  }

  ui->spinBoxInputLayers->setValue(inputCount);
  ui->spinBoxInputNeuronsX->setValue((int)network_params.input_size_x);
  ui->spinBoxInputNeuronsY->setValue((int)network_params.input_size_y);

  ui->spinBoxHiddenLayers->setValue(hiddenCount);
  ui->spinBoxHiddenNeuronsX->setValue((int)network_params.hidden_size_x);
  ui->spinBoxHiddenNeuronsY->setValue((int)network_params.hidden_size_y);

  ui->spinBoxOutputLayers->setValue(outputCount);
  ui->spinBoxOutputNeuronsX->setValue((int)network_params.output_size_x);
  ui->spinBoxOutputNeuronsY->setValue((int)network_params.output_size_y);

  ui->doubleSpinBoxLR->setValue((double)network_params.learning_rate);
  ui->checkBoxLRAdaptive->setChecked(network_params.adaptive_learning_rate);
  ui->doubleSpinBoxLRAdatpiveFactor->setValue(
      (double)network_params.adaptive_learning_rate_factor);
  ui->checkBoxLRAdaptiveIncrease->setChecked(
      network_params.enable_adaptive_increase);

  ui->comboBoxActivationFunctionHidden->setCurrentIndex(
      static_cast<int>(network_params.hidden_activation_function));
  ui->doubleSpinBoxActivationFunctionHiddenAlpha->setValue(
      (double)network_params.hidden_activation_alpha);

  ui->comboBoxActivationFunctionOutput->setCurrentIndex(
      static_cast<int>(network_params.output_activation_function));
  ui->doubleSpinBoxActivationFunctionOutputAlpha->setValue(
      (double)network_params.output_activation_alpha);

  ui->doubleSpinBoxErrorMin->setValue((double)network_params.error_min);
  ui->doubleSpinBoxErrorMax->setValue((double)network_params.error_max);
}

void Bindings::setNetworkToImport(const std::string &value) {
  Manager::getInstance().app_params.network_to_import = value;
}

void Bindings::setActivationFunctionHidden(
    const sipai::EActivationFunction &value) {
  Manager::getInstance().network_params.hidden_activation_function = value;
}

void Bindings::setActivationFunctionOutput(
    const sipai::EActivationFunction &value) {
  Manager::getInstance().network_params.output_activation_function = value;
}

void Bindings::setInputNeuronsX(const int value) {
  Manager::getInstance().network_params.input_size_x = value;
}

void Bindings::setInputNeuronsY(const int value) {
  Manager::getInstance().network_params.input_size_y = value;
}

void Bindings::setHiddenNeuronsX(const int value) {
  Manager::getInstance().network_params.hidden_size_x = value;
}

void Bindings::setHiddenNeuronsY(const int value) {
  Manager::getInstance().network_params.hidden_size_y = value;
}

void Bindings::setOutputNeuronsX(const int value) {
  Manager::getInstance().network_params.output_size_x = value;
}

void Bindings::setOutputNeuronsY(const int value) {
  Manager::getInstance().network_params.output_size_y = value;
}

void Bindings::setRunningMode(const sipai::ERunMode &value) {
  Manager::getInstance().app_params.run_mode = value;
}