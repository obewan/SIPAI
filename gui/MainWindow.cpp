
#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include "./ui_MainWindow.h"
#include "MainWindow.h"
#include "NetworkLoader.h"
#include <sstream>

#include <QFileDialog>
#include <QMessageBox>

using namespace Qt::StringLiterals;
using namespace sipai;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      modelLogger(new QStandardItemModel(0, 3)), progressDialog(nullptr),
      workerThread(nullptr) {
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

  ui->lineEditCurrentNetwork->setText("");

  QFile file(fileName);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("Error"), tr("Cannot open file"));
    return;
  }

  currentFileName = fileName;
  progressDialog =
      new QProgressDialog("Loading neural network...", "Abort", 0, 100, this);
  progressDialog->setWindowModality(Qt::WindowModal);

  NetworkLoader *loader = new NetworkLoader;
  loader->setFileName(fileName);

  workerThread = new QThread;
  loader->moveToThread(workerThread);

  connect(workerThread, &QThread::started, loader, &NetworkLoader::loadNetwork);
  connect(loader, &NetworkLoader::progressUpdated, this,
          &MainWindow::onProgressUpdated, Qt::QueuedConnection);
  connect(loader, &NetworkLoader::loadingFinished, this,
          &MainWindow::onLoadingFinished, Qt::QueuedConnection);
  connect(loader, &NetworkLoader::errorOccurred, this,
          &MainWindow::onErrorOccurred, Qt::QueuedConnection);

  connect(progressDialog, &QProgressDialog::canceled, [loader, this]() {
    loader->deleteLater();
    workerThread->quit();
    workerThread->wait();
    workerThread->deleteLater();
  });

  connect(workerThread, &QThread::finished, loader, &QObject::deleteLater);
  connect(workerThread, &QThread::finished, workerThread,
          &QObject::deleteLater);

  workerThread->start();
}

void MainWindow::onProgressUpdated(int value) {
  if (progressDialog) {
    progressDialog->setValue(value);
  }
}

void MainWindow::onLoadingFinished() {
  if (progressDialog) {
    progressDialog->setValue(100);
    ui->lineEditCurrentNetwork->setText(currentFileName);
    progressDialog->close();
    progressDialog->deleteLater();
    progressDialog = nullptr;
  }
  workerThread->quit();
  workerThread->wait();
  workerThread->deleteLater();
  workerThread = nullptr;
}

void MainWindow::onErrorOccurred(const QString &message) {
  QMetaObject::invokeMethod(
      this,
      [this, message]() {
        if (progressDialog) {
          progressDialog->close();
          progressDialog->deleteLater();
          progressDialog = nullptr;
        }
        QMessageBox::warning(this, tr("Error"), message);
        workerThread->quit();
        workerThread->wait();
        workerThread->deleteLater();
        workerThread = nullptr;
      },
      Qt::QueuedConnection);
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}
