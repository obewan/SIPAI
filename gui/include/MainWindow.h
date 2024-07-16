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

#include <QMainWindow>

QT_BEGIN_NAMESPACE
class QAction;
class QMenu;
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  MainWindow();
};