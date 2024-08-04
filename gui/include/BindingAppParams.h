/**
 * @file BindingAppParams.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Binding for app_params
 * @date 2024-08-02
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once
#include "AppParams.h"
#include "Common.h"
#include <QObject>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class BindingAppParams : public QObject {
  Q_OBJECT
  Q_PROPERTY(int runningMode READ getRunningMode WRITE setRunningMode NOTIFY
                 runningModeChanged)
  Q_PROPERTY(QString networkToImport READ getNetworkToImport WRITE
                 setNetworkToImport NOTIFY networkToImportChanged)
  Q_PROPERTY(QString inputFile READ getInputFile WRITE setInputFile NOTIFY
                 inputFileChanged)
  Q_PROPERTY(QString outputFile READ getOutputFile WRITE setOutputFile NOTIFY
                 outputFileChanged)
  Q_PROPERTY(QString trainingFile READ getTrainingFile WRITE setTrainingFile
                 NOTIFY trainingFileChanged)
  Q_PROPERTY(QString trainingFolder READ getTrainingFolder WRITE
                 setTrainingFolder NOTIFY trainingFolderChanged)
  Q_PROPERTY(QString trainingFolder READ getTrainingFolder WRITE
                 setTrainingFolder NOTIFY trainingFolderChanged)
  Q_PROPERTY(double outputScale READ getOutputScale WRITE setOutputScale NOTIFY
                 outputScaleChanged)
  Q_PROPERTY(double trainingSplitRatio READ getTrainingSplitRatio WRITE
                 setTrainingSplitRatio NOTIFY trainingSplitRatioChanged)
  Q_PROPERTY(double learningRateMin READ getLearningRateMin WRITE
                 setLearningRateMin NOTIFY learningRateMinChanged)
  Q_PROPERTY(double learningRateMax READ getLearningRateMax WRITE
                 setLearningRateMax NOTIFY learningRateMaxChanged)
  Q_PROPERTY(int epochsMax READ getEpochsMax WRITE setEpochsMax NOTIFY
                 epochsMaxChanged)
  Q_PROPERTY(int epochsWithoutImprovementMax READ getEpochsWithoutImprovementMax
                 WRITE setEpochsWithoutImprovementMax NOTIFY
                     epochsWithoutImprovementMaxChanged)
  Q_PROPERTY(int imageSplit READ getImageSplit WRITE setImageSplit NOTIFY
                 imageSplitChanged)
  Q_PROPERTY(double trainingReduceFactor READ getTrainingReduceFactor WRITE
                 setTrainingReduceFactor NOTIFY trainingReduceFactorChanged)

public:
  BindingAppParams(QObject *parent = nullptr);
  void connectUi(Ui::MainWindow *ui);
  void reload();

  int getRunningMode() const { return static_cast<int>(app_params.run_mode); }
  void setRunningMode(int value) {
    auto evalue = static_cast<sipai::ERunMode>(value);
    if (app_params.run_mode != evalue) {
      app_params.run_mode = evalue;
      emit runningModeChanged(value);
    }
  }

  QString getNetworkToImport() const {
    return QString::fromStdString(app_params.network_to_import);
  }
  void setNetworkToImport(const QString &value) {
    if (app_params.network_to_import != value.toStdString()) {
      app_params.network_to_import = value.toStdString();
      emit networkToImportChanged(value);
    }
  }

  QString getInputFile() const {
    return QString::fromStdString(app_params.input_file);
  }
  void setInputFile(const QString &value) {
    if (app_params.input_file != value.toStdString()) {
      app_params.input_file = value.toStdString();
      emit inputFileChanged(value);
    }
  }

  QString getOutputFile() const {
    return QString::fromStdString(app_params.output_file);
  }
  void setOutputFile(const QString &value) {
    if (app_params.output_file != value.toStdString()) {
      app_params.output_file = value.toStdString();
      emit outputFileChanged(value);
    }
  }

  QString getTrainingFile() const {
    return QString::fromStdString(app_params.training_data_file);
  }
  void setTrainingFile(const QString &value) {
    if (app_params.training_data_file != value.toStdString()) {
      app_params.training_data_file = value.toStdString();
      emit trainingFileChanged(value);
    }
  }

  QString getTrainingFolder() const {
    return QString::fromStdString(app_params.training_data_folder);
  }
  void setTrainingFolder(const QString &value) {
    if (app_params.training_data_folder != value.toStdString()) {
      app_params.training_data_folder = value.toStdString();
      emit trainingFolderChanged(value);
    }
  }

  double getOutputScale() const {
    return static_cast<double>(app_params.output_scale);
  }
  void setOutputScale(double value) {
    auto fvalue = static_cast<float>(value);
    if (app_params.output_scale != fvalue) {
      app_params.output_scale = fvalue;
      emit outputScaleChanged(value);
    }
  }

  double getTrainingSplitRatio() const {
    return static_cast<double>(app_params.training_split_ratio);
  }
  void setTrainingSplitRatio(double value) {
    auto fvalue = static_cast<float>(value);
    if (app_params.training_split_ratio != fvalue) {
      app_params.training_split_ratio = fvalue;
      emit trainingSplitRatioChanged(value);
    }
  }

  double getLearningRateMin() const {
    return static_cast<double>(app_params.learning_rate_min);
  }
  void setLearningRateMin(double value) {
    auto fvalue = static_cast<float>(value);
    if (app_params.learning_rate_min != fvalue) {
      app_params.learning_rate_min = fvalue;
      emit learningRateMinChanged(value);
    }
  }

  double getLearningRateMax() const {
    return static_cast<double>(app_params.learning_rate_max);
  }
  void setLearningRateMax(double value) {
    auto fvalue = static_cast<float>(value);
    if (app_params.learning_rate_max != fvalue) {
      app_params.learning_rate_max = fvalue;
      emit learningRateMaxChanged(value);
    }
  }

  int getEpochsMax() const { return static_cast<int>(app_params.max_epochs); }
  void setEpochsMax(int value) {
    auto svalue = static_cast<size_t>(value);
    if (app_params.max_epochs != svalue) {
      app_params.max_epochs = svalue;
      emit epochsMaxChanged(value);
    }
  }

  int getEpochsWithoutImprovementMax() const {
    return static_cast<int>(app_params.max_epochs_without_improvement);
  }
  void setEpochsWithoutImprovementMax(int value) {
    auto svalue = static_cast<size_t>(value);
    if (app_params.max_epochs_without_improvement != svalue) {
      app_params.max_epochs_without_improvement = svalue;
      emit epochsWithoutImprovementMaxChanged(value);
    }
  }

  int getEpochsAutoSave() const {
    return static_cast<int>(app_params.epoch_autosave);
  }
  void setEpochsAutoSave(int value) {
    auto svalue = static_cast<size_t>(value);
    if (app_params.epoch_autosave != svalue) {
      app_params.epoch_autosave = svalue;
      emit epochsAutoSaveChanged(value);
    }
  }

  int getImageSplit() const { return static_cast<int>(app_params.image_split); }
  void setImageSplit(int value) {
    auto svalue = static_cast<size_t>(value);
    if (app_params.image_split != svalue) {
      app_params.image_split = svalue;
      emit imageSplitChanged(value);
    }
  }

  double getTrainingReduceFactor() const {
    return static_cast<double>(app_params.training_reduce_factor);
  }
  void setTrainingReduceFactor(double value) {
    auto fvalue = static_cast<float>(value);
    if (app_params.training_reduce_factor != fvalue) {
      app_params.training_reduce_factor = fvalue;
      emit trainingReduceFactorChanged(value);
    }
  }

signals:
  void runningModeChanged(int value);
  void networkToImportChanged(const QString &value);
  void inputFileChanged(const QString &value);
  void outputFileChanged(const QString &value);
  void trainingFileChanged(const QString &value);
  void trainingFolderChanged(const QString &value);
  void outputScaleChanged(double value);
  void trainingSplitRatioChanged(double value);
  void learningRateMinChanged(double value);
  void learningRateMaxChanged(double value);
  void epochsMaxChanged(int value);
  void epochsWithoutImprovementMaxChanged(int value);
  void epochsAutoSaveChanged(int value);
  void imageSplitChanged(int value);
  void trainingReduceFactorChanged(double value);

private:
  sipai::AppParams &app_params;
};