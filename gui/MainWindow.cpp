#include "MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QFileDialog>
#include <QLibraryInfo>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>

using namespace Qt::StringLiterals;

MainWindow::MainWindow() {

  createActions();
  createMenus();

  setWindowTitle(tr("SIPAI"));
  resize(800, 600);
}

void MainWindow::closeEvent(QCloseEvent *) {}

void MainWindow::about() {
  QMessageBox::about(this, tr("About SIPAI"),
                     tr("Simple Image Processing Artificial Intelligence\n\n"
                        "A Dams-Labs project (www.dams-labs.net)\n\n"
                        "Author: Damien S. Balima\n\n"
                        "Sources: https://obewan.github.io/SIPAI\n"
                        "Copyright: CC BY-NC-SA 4.0"));
}

void MainWindow::modelFileOpen() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      tr("JSON (*.json)"));
  // TODO: continue with check and loading file
}

void MainWindow::createActions() {
  modelFileOpenAct = new QAction(tr("&Import a neural network model..."), this);
  modelFileOpenAct->setShortcut(QKeySequence::Open);
  connect(modelFileOpenAct, &QAction::triggered, this,
          &MainWindow::modelFileOpen);

  exitAct = new QAction(tr("E&xit"), this);
  exitAct->setShortcuts(QKeySequence::Quit);
  connect(exitAct, &QAction::triggered, this, &QWidget::close);

  aboutAct = new QAction(tr("&About"), this);
  connect(aboutAct, &QAction::triggered, this, &MainWindow::about);
}

void MainWindow::createMenus() {
  fileMenu = new QMenu(tr("&File"), this);
  fileMenu->addAction(modelFileOpenAct);
  fileMenu->addSeparator();
  fileMenu->addAction(exitAct);

  helpMenu = new QMenu(tr("&Help"), this);
  helpMenu->addAction(aboutAct);

  menuBar()->addMenu(fileMenu);
  menuBar()->addMenu(helpMenu);
}
