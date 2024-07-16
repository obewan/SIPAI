#include "MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QLibraryInfo>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>

using namespace Qt::StringLiterals;

MainWindow::MainWindow() {
  setWindowTitle(tr("SIPAI"));
  resize(750, 400);
}