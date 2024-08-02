/**
 * @file Bindings.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Parameters bindings for Qt UI
 * @date 2024-08-02
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once

#include "ActivationFunctions.h"
#include <QObject>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class Bindings : public QObject {
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

  Q_INVOKABLE void setInputNeuronsX(const int value);
  Q_INVOKABLE void setInputNeuronsY(const int value);
  Q_INVOKABLE void setHiddenNeuronsX(const int value);
  Q_INVOKABLE void setHiddenNeuronsY(const int value);
  Q_INVOKABLE void setOutputNeuronsX(const int value);
  Q_INVOKABLE void setOutputNeuronsY(const int value);
};