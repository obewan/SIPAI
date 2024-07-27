
#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include "./ui_MainWindow.h"
#include "MainWindow.h"
#include <sstream>

#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>

using namespace Qt::StringLiterals;
using namespace sipai;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      modelLogger(new QStandardItemModel(0, 3)), progressDialog(nullptr),
      futureWatcher(new QFutureWatcher<void>(this)) {
  // Setup the UI with the MainWindow.ui
  ui->setupUi(this);

  // Connect actions to slots
  connect(ui->actionLoadNeuralNetwork, &QAction::triggered, this,
          &MainWindow::onActionLoadNeuralNetwork);
  connect(ui->actionAbout, &QAction::triggered, this,
          &MainWindow::onActionAbout);

  // Connect the QFutureWatcher signals to appropriate slots
  connect(futureWatcher, &QFutureWatcher<void>::finished, this,
          &MainWindow::onLoadingFinished);
  connect(futureWatcher, &QFutureWatcher<void>::progressValueChanged, this,
          &MainWindow::onProgressUpdated);

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

  connect(progressDialog, &QProgressDialog::canceled, this,
          &MainWindow::onLoadingCanceled);

  // Update status bar
  statusBar()->showMessage(tr("Loading neural network..."));

  // Start the concurrent loading process
  QFuture<void> future = QtConcurrent::run([this]() { loadNetwork(); });
  futureWatcher->setFuture(future);

  progressDialog->show();
}

void MainWindow::onProgressUpdated(int value) {
  if (progressDialog) {
    progressDialog->setValue(value);
  }
}

void MainWindow::onLoadingCanceled() {
  if (futureWatcher->isRunning()) {
    futureWatcher->cancel();
  }
  if (progressDialog) {
    progressDialog->close();
    progressDialog->deleteLater();
  }
  statusBar()->showMessage(tr("Loading canceled"),
                           5000); // Show message for 5 seconds
}

void MainWindow::onLoadingFinished() {
  if (progressDialog) {
    progressDialog->setValue(100);
    ui->lineEditCurrentNetwork->setText(currentFileName);
    progressDialog->close();
    progressDialog->deleteLater();
  }
  statusBar()->showMessage(tr("Loading finished"),
                           5000); // Show message for 5 seconds
}

void MainWindow::onErrorOccurred(const QString &message) {
  QMetaObject::invokeMethod(
      this,
      [this, message]() {
        if (progressDialog) {
          progressDialog->close();
          progressDialog->deleteLater();
        }
        statusBar()->showMessage(tr("Error: %1").arg(message),
                                 5000); // Show message for 5 seconds
        QMessageBox::warning(this, tr("Error"), message);
      },
      Qt::QueuedConnection);
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}

void MainWindow::loadNetwork() {
  auto &manager = Manager::getInstance();
  manager.app_params.network_to_import = currentFileName.toStdString();

  try {
    manager.createOrImportNetwork([this](int i) {
      QMetaObject::invokeMethod(futureWatcher, "setProgressValue",
                                Q_ARG(int, i));
      // Check for cancellation
      if (futureWatcher->isCanceled()) {
        throw std::runtime_error("Loading canceled");
      }
    });
  } catch (const std::exception &ex) {
    QMetaObject::invokeMethod(this, "onErrorOccurred",
                              Q_ARG(QString, ex.what()));
  }
}