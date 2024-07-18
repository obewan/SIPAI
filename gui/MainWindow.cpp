
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include "MainWindow.h"
#include "./ui_MainWindow.h"
#include "Manager.h"
#include <sstream>

#include <QFileDialog>
#include <QMessageBox>

using namespace Qt::StringLiterals;
using namespace sipai;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  // Setup the UI with the MainWindow.ui
  ui->setupUi(this);

  // Connect to slots
  connect(ui->actionImportNeuralNetwork, &QAction::triggered, this,
          &MainWindow::onActionImportNeuralNetwork);
  connect(ui->actionAbout, &QAction::triggered, this,
          &MainWindow::onActionAbout);

  // Other inits
  const std::string &version = Manager::getConstInstance().app_params.version;
  std::stringstream aboutStr;
  aboutStr << "Simple Image Processing Artificial Intelligence\n"
           << "Version: " << version << "\n\n"
           << "A Dams-Labs project (www.dams-labs.net)\n"
           << "Author: Damien S. Balima\n"
           << "Sources: https://obewan.github.io/SIPAI\n"
           << "Copyright: CC BY-NC-SA 4.0";
  aboutStr_ = aboutStr.str();
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::onActionImportNeuralNetwork() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      tr("JSON (*.json)"));
  // TODO: continue with check and loading file
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}
