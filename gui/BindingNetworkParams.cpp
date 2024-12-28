#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "BindingNetworkParams.h"
#include "MainWindow.h"

#include <algorithm>
#include <ranges>

using namespace Qt::StringLiterals;
using namespace sipai;

BindingNetworkParams::BindingNetworkParams(QObject *parent)
    : QObject(parent), network_params(Manager::getInstance().network_params) {}

void BindingNetworkParams::connectUi(Ui::MainWindow *ui) {
  // add activation functions enums selects
  for (const auto &[key, value] : activation_map) {
    ui->comboBoxActivationFunctionHidden->addItem(QString::fromStdString(key),
                                                  static_cast<int>(value));
    ui->comboBoxActivationFunctionOutput->addItem(QString::fromStdString(key),
                                                  static_cast<int>(value));
  }

  // bindings
  connect(ui->comboBoxActivationFunctionHidden,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &BindingNetworkParams::setActivationFunctionHidden);
  connect(this, &BindingNetworkParams::activationFunctionHiddenChanged,
          [ui](int value) {
            ui->comboBoxActivationFunctionHidden->setCurrentIndex(value);
          });

  connect(ui->comboBoxActivationFunctionOutput,
          QOverload<int>::of(&QComboBox::currentIndexChanged), this,
          &BindingNetworkParams::setActivationFunctionOutput);
  connect(this, &BindingNetworkParams::activationFunctionOutputChanged,
          [ui](int value) {
            ui->comboBoxActivationFunctionOutput->setCurrentIndex(value);
          });

  connect(ui->doubleSpinBoxActivationFunctionHiddenAlpha,
          &QDoubleSpinBox::valueChanged, this,
          &BindingNetworkParams::setActivationAlphaHidden);
  connect(this, &BindingNetworkParams::activationAlphaHiddenChanged,
          [ui](double value) {
            ui->doubleSpinBoxActivationFunctionHiddenAlpha->setValue(value);
          });

  connect(ui->doubleSpinBoxActivationFunctionOutputAlpha,
          &QDoubleSpinBox::valueChanged, this,
          &BindingNetworkParams::setActivationAlphaOutput);
  connect(this, &BindingNetworkParams::activationAlphaOutputChanged,
          [ui](double value) {
            ui->doubleSpinBoxActivationFunctionOutputAlpha->setValue(value);
          });

  connect(ui->spinBoxHiddenLayers, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setHiddenLayersCount);
  connect(this, &BindingNetworkParams::hiddenLayersCountChanged,
          [ui](int value) { ui->spinBoxHiddenLayers->setValue(value); });

  connect(ui->spinBoxInputNeuronsX, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setInputNeuronsX);
  connect(this, &BindingNetworkParams::inputNeuronsXChanged,
          [ui](int value) { ui->spinBoxInputNeuronsX->setValue(value); });

  connect(ui->spinBoxInputNeuronsY, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setInputNeuronsY);
  connect(this, &BindingNetworkParams::inputNeuronsYChanged,
          [ui](int value) { ui->spinBoxInputNeuronsY->setValue(value); });

  connect(ui->spinBoxHiddenNeuronsX, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setHiddenNeuronsX);
  connect(this, &BindingNetworkParams::hiddenNeuronsXChanged,
          [ui](int value) { ui->spinBoxHiddenNeuronsX->setValue(value); });

  connect(ui->spinBoxHiddenNeuronsY, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setHiddenNeuronsY);
  connect(this, &BindingNetworkParams::hiddenNeuronsYChanged,
          [ui](int value) { ui->spinBoxHiddenNeuronsY->setValue(value); });

  connect(ui->spinBoxOutputNeuronsX, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setOutputNeuronsX);
  connect(this, &BindingNetworkParams::outputNeuronsXChanged,
          [ui](int value) { ui->spinBoxOutputNeuronsX->setValue(value); });

  connect(ui->spinBoxOutputNeuronsY, &QSpinBox::valueChanged, this,
          &BindingNetworkParams::setOutputNeuronsY);
  connect(this, &BindingNetworkParams::outputNeuronsYChanged,
          [ui](int value) { ui->spinBoxOutputNeuronsY->setValue(value); });

  connect(ui->doubleSpinBoxLR, &QDoubleSpinBox::valueChanged, this,
          &BindingNetworkParams::setLearningRate);
  connect(this, &BindingNetworkParams::learningRateChanged,
          [ui](double value) { ui->doubleSpinBoxLR->setValue(value); });

  connect(ui->checkBoxLRAdaptive, &QCheckBox::toggled, this,
          &BindingNetworkParams::setAdaptiveLearningRate);
  connect(this, &BindingNetworkParams::adaptiveLearningRateChanged,
          [ui](bool value) { ui->checkBoxLRAdaptive->setChecked(value); });

  connect(ui->doubleSpinBoxLRAdatpiveFactor, &QDoubleSpinBox::valueChanged,
          this, &BindingNetworkParams::setAdaptiveLearningRateFactor);
  connect(this, &BindingNetworkParams::adaptiveLearningRateFactorChanged,
          [ui](double value) {
            ui->doubleSpinBoxLRAdatpiveFactor->setValue(value);
          });

  connect(ui->checkBoxLRAdaptiveIncrease, &QCheckBox::toggled, this,
          &BindingNetworkParams::setAdaptiveLearningRateIncrease);
  connect(
      this, &BindingNetworkParams::adaptiveLearningRateIncreaseChanged,
      [ui](bool value) { ui->checkBoxLRAdaptiveIncrease->setChecked(value); });

  connect(ui->doubleSpinBoxErrorMin, &QDoubleSpinBox::valueChanged, this,
          &BindingNetworkParams::setErrorMin);
  connect(this, &BindingNetworkParams::errorMinChanged,
          [ui](double value) { ui->doubleSpinBoxErrorMin->setValue(value); });

  connect(ui->doubleSpinBoxErrorMax, &QDoubleSpinBox::valueChanged, this,
          &BindingNetworkParams::setErrorMax);
  connect(this, &BindingNetworkParams::errorMaxChanged,
          [ui](double value) { ui->doubleSpinBoxErrorMax->setValue(value); });
}

void BindingNetworkParams::reload() {
  emit activationFunctionHiddenChanged(getActivationFunctionHidden());
  emit activationFunctionOutputChanged(getActivationFunctionOutput());
  emit activationAlphaHiddenChanged(getActivationAlphaHidden());
  emit activationAlphaOutputChanged(getActivationAlphaOutput());
  emit hiddenLayersCountChanged(getHiddenLayersCount());

  emit inputNeuronsXChanged(getInputNeuronsX());
  emit inputNeuronsYChanged(getInputNeuronsY());
  emit hiddenNeuronsXChanged(getHiddenNeuronsX());
  emit hiddenNeuronsYChanged(getHiddenNeuronsY());
  emit outputNeuronsXChanged(getOutputNeuronsX());
  emit outputNeuronsYChanged(getOutputNeuronsY());

  emit learningRateChanged(getLearningRate());
  emit adaptiveLearningRateChanged(getAdaptiveLearningRate());
  emit adaptiveLearningRateFactorChanged(getAdaptiveLearningRateFactor());
  emit adaptiveLearningRateIncreaseChanged(getAdaptiveLearningRateIncrease());
  emit errorMinChanged(getErrorMin());
  emit errorMaxChanged(getErrorMax());
}
