#include "MainWindow.h"
#include "./ui_MainWindow.h"

#include <QFileDialog>
#include <QMessageBox>

using namespace Qt::StringLiterals;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  connect(ui->actionImportNeuralNetwork, &QAction::triggered, this,
          &MainWindow::onActionImportNeuralNetwork);
  connect(ui->actionAbout, &QAction::triggered, this,
          &MainWindow::onActionAbout);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::onActionImportNeuralNetwork() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      tr("JSON (*.json)"));
  // TODO: continue with check and loading file
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"),
                     tr("Simple Image Processing Artificial Intelligence\n\n"
                        "A Dams-Labs project (www.dams-labs.net)\n\n"
                        "Author: Damien S. Balima\n\n"
                        "Sources: https://obewan.github.io/SIPAI\n"
                        "Copyright: CC BY-NC-SA 4.0"));
}
