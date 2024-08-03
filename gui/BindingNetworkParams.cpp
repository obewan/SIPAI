#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "BindingNetworkParams.h"
#include "MainWindow.h"

#include <algorithm>
#include <ranges>

using namespace Qt::StringLiterals;
using namespace sipai;

void BindingNetworkParams::setBindings(Ui::MainWindow *ui) {
  // bind some params
  connect(
      ui->lineEditCurrentNetwork, &QLineEdit::textChanged, this,
      [this](const QString &text) { setNetworkToImport(text.toStdString()); });
  connect(ui->spinBoxInputNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setInputNeuronsX((size_t)value); });
  connect(ui->spinBoxInputNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setInputNeuronsY((size_t)value); });
  connect(ui->spinBoxHiddenNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setHiddenNeuronsX((size_t)value); });
  connect(ui->spinBoxHiddenNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setHiddenNeuronsY((size_t)value); });
  connect(ui->spinBoxOutputNeuronsX, &QSpinBox::valueChanged, this,
          [this](int value) { setOutputNeuronsX((size_t)value); });
  connect(ui->spinBoxOutputNeuronsY, &QSpinBox::valueChanged, this,
          [this](int value) { setOutputNeuronsY((size_t)value); });
  connect(ui->doubleSpinBoxErrorMin, &QDoubleSpinBox::valueChanged, this,
          [this](double value) { setErrorMin((float)value); });
  connect(ui->doubleSpinBoxErrorMax, &QDoubleSpinBox::valueChanged, this,
          [this](double value) { setErrorMax((float)value); });

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
  // Some other running parameters bindings
  connect(ui->lineEditInputFile, &QLineEdit::textChanged, this,
          [this](const QString &text) { setInputFile(text.toStdString()); });
  connect(ui->lineEditOutputFile, &QLineEdit::textChanged, this,
          [this](const QString &text) { setOutputFile(text.toStdString()); });
  connect(ui->lineEditTrainingFile, &QLineEdit::textChanged, this,
          [this](const QString &text) { setTrainingFile(text.toStdString()); });
  connect(
      ui->lineEditTrainingFolder, &QLineEdit::textChanged, this,
      [this](const QString &text) { setTrainingFolder(text.toStdString()); });
  connect(ui->doubleSpinBoxOutputScaleFactor, &QDoubleSpinBox::valueChanged,
          this, [this](double value) { setOutputScale((float)value); });
  connect(ui->doubleSpinBoxTrainingSplit, &QDoubleSpinBox::valueChanged, this,
          [this](double value) { setTrainingSplitRatio((float)value); });
  connect(ui->doubleSpinBoxLearningRateMin, &QDoubleSpinBox::valueChanged, this,
          [this](double value) { setLearningRateMin((float)value); });
  connect(ui->doubleSpinBoxLearningRateMax, &QDoubleSpinBox::valueChanged, this,
          [this](double value) { setLearningRateMax((float)value); });
  connect(ui->spinBoxEpochsMax, &QSpinBox::valueChanged, this,
          [this](int value) { setEpochsMax((size_t)value); });
  connect(ui->spinBoxEpochsWithoutImpMax, &QSpinBox::valueChanged, this,
          [this](int value) { setEpochsWithoutImprovementMax((size_t)value); });
  connect(ui->spinBoxEpochsSaving, &QSpinBox::valueChanged, this,
          [this](int value) { setEpochsAutoSave((size_t)value); });
}

void BindingNetworkParams::getAppParams(Ui::MainWindow *ui) {
  const auto &app_params = Manager::getConstInstance().app_params;
  ui->lineEditCurrentNetwork->setText(app_params.network_to_import.c_str());

  ui->comboBoxMode->setCurrentIndex(static_cast<int>(app_params.run_mode));
  ui->lineEditInputFile->setText(app_params.input_file.c_str());
  ui->lineEditOutputFile->setText(app_params.output_file.c_str());
  ui->lineEditTrainingFile->setText(app_params.training_data_file.c_str());
  ui->lineEditTrainingFolder->setText(app_params.training_data_folder.c_str());

  ui->doubleSpinBoxOutputScaleFactor->setValue((double)app_params.output_scale);
  ui->doubleSpinBoxTrainingSplit->setValue(
      (double)app_params.training_split_ratio);
  ui->doubleSpinBoxLearningRateMin->setValue(
      (double)app_params.learning_rate_min);
  ui->doubleSpinBoxLearningRateMax->setValue(
      (double)app_params.learning_rate_max);
  ui->spinBoxEpochsMax->setValue((int)app_params.max_epochs);
  ui->spinBoxEpochsWithoutImpMax->setValue(
      (int)app_params.max_epochs_without_improvement);
  ui->spinBoxEpochsSaving->setValue((int)app_params.epoch_autosave);
}

void BindingNetworkParams::getNetworkParams(Ui::MainWindow *ui) {
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

void BindingNetworkParams::setNetworkToImport(const std::string &value) {
  Manager::getInstance().app_params.network_to_import = value;
}

void BindingNetworkParams::setActivationFunctionHidden(
    const sipai::EActivationFunction &value) {
  Manager::getInstance().network_params.hidden_activation_function = value;
}

void BindingNetworkParams::setActivationFunctionOutput(
    const sipai::EActivationFunction &value) {
  Manager::getInstance().network_params.output_activation_function = value;
}

void BindingNetworkParams::setInputNeuronsX(const size_t value) {
  Manager::getInstance().network_params.input_size_x = value;
}

void BindingNetworkParams::setInputNeuronsY(const size_t value) {
  Manager::getInstance().network_params.input_size_y = value;
}

void BindingNetworkParams::setHiddenNeuronsX(const size_t value) {
  Manager::getInstance().network_params.hidden_size_x = value;
}

void BindingNetworkParams::setHiddenNeuronsY(const size_t value) {
  Manager::getInstance().network_params.hidden_size_y = value;
}

void BindingNetworkParams::setOutputNeuronsX(const size_t value) {
  Manager::getInstance().network_params.output_size_x = value;
}

void BindingNetworkParams::setOutputNeuronsY(const size_t value) {
  Manager::getInstance().network_params.output_size_y = value;
}

void BindingNetworkParams::setErrorMin(const float value) {
  Manager::getInstance().network_params.error_min = value;
}

void BindingNetworkParams::setErrorMax(const float value) {
  Manager::getInstance().network_params.error_max = value;
}

void BindingNetworkParams::setRunningMode(const sipai::ERunMode &value) {
  Manager::getInstance().app_params.run_mode = value;
}

void BindingNetworkParams::setInputFile(const std::string &value) {
  Manager::getInstance().app_params.input_file = value;
}

void BindingNetworkParams::setOutputFile(const std::string &value) {
  Manager::getInstance().app_params.output_file = value;
}

void BindingNetworkParams::setTrainingFile(const std::string &value) {
  Manager::getInstance().app_params.training_data_file = value;
}

void BindingNetworkParams::setTrainingFolder(const std::string &value) {
  Manager::getInstance().app_params.training_data_folder = value;
}

void BindingNetworkParams::setOutputScale(const float value) {
  Manager::getInstance().app_params.output_scale = value;
}

void BindingNetworkParams::setTrainingSplitRatio(const float value) {
  Manager::getInstance().app_params.training_split_ratio = value;
}

void BindingNetworkParams::setLearningRateMin(const float value) {
  Manager::getInstance().app_params.learning_rate_min = value;
}

void BindingNetworkParams::setLearningRateMax(const float value) {
  Manager::getInstance().app_params.learning_rate_max = value;
}

void BindingNetworkParams::setEpochsMax(const size_t value) {
  Manager::getInstance().app_params.max_epochs = value;
}

void BindingNetworkParams::setEpochsWithoutImprovementMax(const size_t value) {
  Manager::getInstance().app_params.max_epochs_without_improvement = value;
}

void BindingNetworkParams::setEpochsAutoSave(const size_t value) {
  Manager::getInstance().app_params.epoch_autosave = value;
}