/**
 * @file MainWindow.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Main Window QT6
 * @date 2024-07-16
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#pragma once

#include "BindingAppParams.h"
#include <QFutureWatcher>
#include <QMainWindow>
#include <QProgressDialog>
#include <QStandardItemModel>
#include <QtConcurrent>
#include <string>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

public slots:
  void onActionAbout();
  void onActionLoadNeuralNetwork();
  void onActionSelectInputFile();
  void onActionSelectOutputFile();
  void onActionSelectTrainingFile();
  void onActionSelectTrainingFolder();

private slots:
  void onProgressUpdated(int value);
  void onLoadingCanceled();
  void onLoadingFinished();
  void onErrorOccurred(const QString &message);

private:
  Ui::MainWindow *ui;
  QStandardItemModel *modelLogger;
  QProgressDialog *progressDialog;
  QFutureWatcher<void> *futureWatcher;
  BindingAppParams *bindingAppParams;

  std::string aboutStr_;
  void loadNetwork();
};