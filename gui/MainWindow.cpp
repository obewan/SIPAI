
#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include "./ui_MainWindow.h"
#include "MainWindow.h"
#include <sstream>

#include <QFileDialog>
#include <QMessageBox>
#include <QProgressDialog>
#include <QThread>

using namespace Qt::StringLiterals;
using namespace sipai;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      modelLogger(new QStandardItemModel(0, 3)) {
  // Setup the UI with the MainWindow.ui
  ui->setupUi(this);

  // Connect actions to slots
  connect(ui->actionLoadNeuralNetwork, &QAction::triggered, this,
          &MainWindow::onActionLoadNeuralNetwork);
  connect(ui->actionAbout, &QAction::triggered, this,
          &MainWindow::onActionAbout);

  // Add logs
  modelLogger->setHorizontalHeaderLabels({"Timestamp", "Log Level", "Message"});
  ui->tableViewLogs->setModel(modelLogger);
  ui->tableViewLogs->horizontalHeader()->setSectionResizeMode(
      0, QHeaderView::ResizeToContents); // Timestamp
  ui->tableViewLogs->horizontalHeader()->setSectionResizeMode(
      1, QHeaderView::ResizeToContents); // Log Level
  ui->tableViewLogs->horizontalHeader()->setSectionResizeMode(
      2, QHeaderView::Stretch); // Message

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

MainWindow::~MainWindow() {
  delete ui;
  delete modelLogger;
}

void MainWindow::onActionLoadNeuralNetwork() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      tr("JSON (*.json)"));

  if (fileName.isEmpty()) {
    return; // No file selected
  }
  Manager::getInstance().app_params.network_to_import = fileName.toStdString();

  QFile file(fileName);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("Error"), tr("Cannot open file"));
    return;
  }

  QProgressDialog progress("Loading neural network...", "Abort", 0, 100, this);
  progress.setWindowModality(Qt::WindowModal);
  for (int i = 0; i < 100; i++) {
    progress.setValue(i);

    if (progress.wasCanceled())
      break;
    //... load here
    // Simulate some loading work
    QThread::msleep(50);
  }
  progress.setValue(100);

  ui->lineEditCurrentNetwork->setText(fileName);
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}
