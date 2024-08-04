#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "BindingAppParams.h"

using namespace Qt::StringLiterals;
using namespace sipai;

BindingAppParams::BindingAppParams(QObject *parent)
    : QObject(parent), app_params(Manager::getInstance().app_params) {}

void BindingAppParams::connectUi(Ui::MainWindow *ui) {
  // add run mode enums select
  for (const auto &[key, value] : mode_map) {
    ui->comboBoxMode->addItem(QString::fromStdString(key),
                              static_cast<int>(value));
  }

  // bindings
  connect(ui->comboBoxMode, QOverload<int>::of(&QComboBox::currentIndexChanged),
          this, &BindingAppParams::setRunningMode);
  connect(this, &BindingAppParams::runningModeChanged,
          [ui](int value) { ui->comboBoxMode->setCurrentIndex(value); });

  connect(ui->lineEditCurrentNetwork, &QLineEdit::textChanged, this,
          &BindingAppParams::setNetworkToImport);
  connect(this, &BindingAppParams::networkToImportChanged,
          [ui](const QString &value) {
            ui->lineEditCurrentNetwork->setText(value);
          });

  connect(ui->lineEditInputFile, &QLineEdit::textChanged, this,
          &BindingAppParams::setInputFile);
  connect(
      this, &BindingAppParams::inputFileChanged,
      [ui](const QString &value) { ui->lineEditInputFile->setText(value); });

  connect(ui->lineEditOutputFile, &QLineEdit::textChanged, this,
          &BindingAppParams::setOutputFile);
  connect(
      this, &BindingAppParams::outputFileChanged,
      [ui](const QString &value) { ui->lineEditOutputFile->setText(value); });

  connect(ui->lineEditTrainingFile, &QLineEdit::textChanged, this,
          &BindingAppParams::setTrainingFile);
  connect(
      this, &BindingAppParams::trainingFileChanged,
      [ui](const QString &value) { ui->lineEditTrainingFile->setText(value); });

  connect(ui->lineEditTrainingFolder, &QLineEdit::textChanged, this,
          &BindingAppParams::setTrainingFolder);
  connect(this, &BindingAppParams::trainingFolderChanged,
          [ui](const QString &value) {
            ui->lineEditTrainingFolder->setText(value);
          });

  connect(ui->doubleSpinBoxOutputScaleFactor, &QDoubleSpinBox::valueChanged,
          this, &BindingAppParams::setOutputScale);
  connect(this, &BindingAppParams::outputScaleChanged, [ui](double value) {
    ui->doubleSpinBoxOutputScaleFactor->setValue(value);
  });

  connect(ui->doubleSpinBoxTrainingSplit, &QDoubleSpinBox::valueChanged, this,
          &BindingAppParams::setTrainingSplitRatio);
  connect(
      this, &BindingAppParams::trainingSplitRatioChanged,
      [ui](double value) { ui->doubleSpinBoxTrainingSplit->setValue(value); });

  connect(ui->doubleSpinBoxLearningRateMin, &QDoubleSpinBox::valueChanged, this,
          &BindingAppParams::setLearningRateMin);
  connect(this, &BindingAppParams::learningRateMinChanged, [ui](double value) {
    ui->doubleSpinBoxLearningRateMin->setValue(value);
  });

  connect(ui->doubleSpinBoxLearningRateMax, &QDoubleSpinBox::valueChanged, this,
          &BindingAppParams::setLearningRateMax);
  connect(this, &BindingAppParams::learningRateMaxChanged, [ui](double value) {
    ui->doubleSpinBoxLearningRateMax->setValue(value);
  });

  connect(ui->spinBoxEpochsMax, &QSpinBox::valueChanged, this,
          &BindingAppParams::setEpochsMax);
  connect(this, &BindingAppParams::epochsMaxChanged,
          [ui](int value) { ui->spinBoxEpochsMax->setValue(value); });

  connect(ui->spinBoxEpochsWithoutImpMax, &QSpinBox::valueChanged, this,
          &BindingAppParams::setEpochsWithoutImprovementMax);
  connect(this, &BindingAppParams::epochsWithoutImprovementMaxChanged,
          [ui](int value) { ui->spinBoxEpochsWithoutImpMax->setValue(value); });

  connect(ui->spinBoxEpochsSaving, &QSpinBox::valueChanged, this,
          &BindingAppParams::setEpochsAutoSave);
  connect(this, &BindingAppParams::epochsAutoSaveChanged,
          [ui](int value) { ui->spinBoxEpochsSaving->setValue(value); });

  connect(ui->spinBoxImageSplits, &QSpinBox::valueChanged, this,
          &BindingAppParams::setImageSplit);
  connect(this, &BindingAppParams::imageSplitChanged,
          [ui](int value) { ui->spinBoxImageSplits->setValue(value); });

  connect(ui->doubleSpinBoxImageReducingFactor, &QDoubleSpinBox::valueChanged,
          this, &BindingAppParams::setTrainingReduceFactor);
  connect(this, &BindingAppParams::trainingReduceFactorChanged,
          [ui](double value) {
            ui->doubleSpinBoxImageReducingFactor->setValue(value);
          });

  connect(ui->checkBoxImagesRandomLoading, &QCheckBox::toggled, this,
          &BindingAppParams::setRandomLoading);
  connect(this, &BindingAppParams::randomLoadingChanged, [ui](bool value) {
    ui->checkBoxImagesRandomLoading->setChecked(value);
  });

  connect(ui->checkBoxImagesBulk, &QCheckBox::toggled, this,
          &BindingAppParams::setBulkLoading);
  connect(this, &BindingAppParams::bulkLoadingChanged,
          [ui](bool value) { ui->checkBoxImagesBulk->setChecked(value); });

  connect(ui->checkBoxImagesPadding, &QCheckBox::toggled, this,
          &BindingAppParams::setImagePadding);
  connect(this, &BindingAppParams::imagePaddingChanged,
          [ui](bool value) { ui->checkBoxImagesPadding->setChecked(value); });

  connect(ui->checkBoxEnableCpuParallel, &QCheckBox::toggled, this,
          &BindingAppParams::setCpuParallel);
  connect(this, &BindingAppParams::cpuParallelChanged, [ui](bool value) {
    ui->checkBoxEnableCpuParallel->setChecked(value);
  });

  connect(ui->checkBoxEnableGpuParallel, &QCheckBox::toggled, this,
          &BindingAppParams::setEnableVulkan);
  connect(this, &BindingAppParams::enableVulkanChanged, [ui](bool value) {
    ui->checkBoxEnableGpuParallel->setChecked(value);
  });

  connect(ui->checkBoxVerbose, &QCheckBox::toggled, this,
          &BindingAppParams::setVerbose);
  connect(this, &BindingAppParams::verboseChanged,
          [ui](bool value) { ui->checkBoxVerbose->setChecked(value); });

  connect(ui->checkBoxVerboseDebug, &QCheckBox::toggled, this,
          &BindingAppParams::setVerboseDebug);
  connect(this, &BindingAppParams::verboseDebugChanged,
          [ui](bool value) { ui->checkBoxVerboseDebug->setChecked(value); });

  connect(ui->checkBoxVerboseVulkan, &QCheckBox::toggled, this,
          &BindingAppParams::setVulkanDebug);
  connect(this, &BindingAppParams::vulkanDebugChanged,
          [ui](bool value) { ui->checkBoxVerboseVulkan->setChecked(value); });
}

void BindingAppParams::reload() {
  emit runningModeChanged(getRunningMode());
  emit networkToImportChanged(getNetworkToImport());
  emit inputFileChanged(getInputFile());
  emit outputFileChanged(getOutputFile());
  emit trainingFileChanged(getTrainingFile());
  emit trainingFolderChanged(getTrainingFolder());
  emit outputScaleChanged(getOutputScale());
  emit trainingSplitRatioChanged(getTrainingSplitRatio());
  emit learningRateMinChanged(getLearningRateMin());
  emit learningRateMaxChanged(getLearningRateMax());
  emit epochsMaxChanged(getEpochsMax());
  emit epochsWithoutImprovementMaxChanged(getEpochsWithoutImprovementMax());
  emit epochsAutoSaveChanged(getEpochsAutoSave());

  emit imageSplitChanged(getImageSplit());
  emit trainingReduceFactorChanged(getTrainingReduceFactor());
  emit randomLoadingChanged(getRandomLoading());
  emit bulkLoadingChanged(getBulkLoading());
  emit imagePaddingChanged(getImagePadding());
  emit cpuParallelChanged(getCpuParallel());
  emit enableVulkanChanged(getEnableVulkan());
  emit verboseChanged(getVerbose());
  emit verboseDebugChanged(getVerboseDebug());
  emit vulkanDebugChanged(getVulkanDebug());
}