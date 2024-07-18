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

#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include <QMainWindow>
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
  void onActionImportNeuralNetwork();

private:
  Ui::MainWindow *ui;
  std::string aboutStr_;
};