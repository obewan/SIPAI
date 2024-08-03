/**
 * @file BindingNetworkParams.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Bindings for network_params
 * @date 2024-08-03
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#include "ActivationFunctions.h"
#include "Common.h"
#include <QObject>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class BindingNetworkParams : public QObject {
  Q_OBJECT

public:
  void setBindings(Ui::MainWindow *ui);
  void getAppParams(Ui::MainWindow *ui);
  void getNetworkParams(Ui::MainWindow *ui);

private:
  Q_INVOKABLE void setNetworkToImport(const std::string &value);
  Q_INVOKABLE void
  setActivationFunctionHidden(const sipai::EActivationFunction &value);
  Q_INVOKABLE void
  setActivationFunctionOutput(const sipai::EActivationFunction &value);

  Q_INVOKABLE void setInputNeuronsX(const size_t value);
  Q_INVOKABLE void setInputNeuronsY(const size_t value);
  Q_INVOKABLE void setHiddenNeuronsX(const size_t value);
  Q_INVOKABLE void setHiddenNeuronsY(const size_t value);
  Q_INVOKABLE void setOutputNeuronsX(const size_t value);
  Q_INVOKABLE void setOutputNeuronsY(const size_t value);
  Q_INVOKABLE void setErrorMin(const float value);
  Q_INVOKABLE void setErrorMax(const float value);

  Q_INVOKABLE void setRunningMode(const sipai::ERunMode &value);
  Q_INVOKABLE void setInputFile(const std::string &value);
  Q_INVOKABLE void setOutputFile(const std::string &value);
  Q_INVOKABLE void setTrainingFile(const std::string &value);
  Q_INVOKABLE void setTrainingFolder(const std::string &value);
  Q_INVOKABLE void setOutputScale(const float value);
  Q_INVOKABLE void setTrainingSplitRatio(const float value);
  Q_INVOKABLE void setLearningRateMin(const float value);
  Q_INVOKABLE void setLearningRateMax(const float value);
  Q_INVOKABLE void setEpochsMax(const size_t value);
  Q_INVOKABLE void setEpochsWithoutImprovementMax(const size_t value);
  Q_INVOKABLE void setEpochsAutoSave(const size_t value);
};