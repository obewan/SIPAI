
#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "MainWindow.h"
#include "QtSimpleLogger.h"
#include "SimpleLogger.h"
#include <sstream>

#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>

using namespace Qt::StringLiterals;
using namespace sipai;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow),
      modelLogger(new QStandardItemModel(0, 3)), progressDialog(nullptr),
      futureWatcher(new QFutureWatcher<void>(this)),
      bindingAppParams(new BindingAppParams()),
      bindingNetworkParams(new BindingNetworkParams()),
      qtSimpleLogger(new QtSimpleLogger(modelLogger)) {

  auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;

  // Setup the UI with the MainWindow.ui
  ui->setupUi(this);

  // Bindings
  bindingAppParams->connectUi(ui);
  bindingNetworkParams->connectUi(ui);

  // Get default values
  bindingAppParams->reload();
  bindingNetworkParams->reload();

  // Connect actions to slots
  connect(ui->actionLoadNeuralNetwork, &QAction::triggered, this,
          &MainWindow::onActionLoadNeuralNetwork);
  connect(ui->actionAbout, &QAction::triggered, this,
          &MainWindow::onActionAbout);
  connect(ui->actionSelectInputFile, &QAction::triggered, this,
          &MainWindow::onActionSelectInputFile);
  connect(ui->actionSelectOutputFile, &QAction::triggered, this,
          &MainWindow::onActionSelectOutputFile);
  connect(ui->actionSelectTrainingFile, &QAction::triggered, this,
          &MainWindow::onActionSelectTrainingFile);
  connect(ui->actionSelectTrainingFolder, &QAction::triggered, this,
          &MainWindow::onActionSelectTrainingFolder);

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

  SimpleLogger &logger =
      const_cast<SimpleLogger &>(SimpleLogger::getInstance());
  logger.setLogCallback([this](const std::string &timestamp,
                               const std::string &level,
                               const std::string &message) {
    qtSimpleLogger->log(timestamp, level, message);
  });

  // Other inits
  const std::string &version = manager.app_params.version;
  std::stringstream aboutStr;
  aboutStr << "Simple Image Processing Artificial Intelligence\n"
           << "Version: " << version << "\n\n"
           << "A Dams-Labs project (www.dams-labs.net)\n"
           << "Author: Damien S. Balima\n"
           << "Sources: https://obewan.github.io/SIPAI\n"
           << "Copyright: CC BY-NC-SA 4.0";
  aboutStr_ = aboutStr.str();

  manager.showHeader();
}

MainWindow::~MainWindow() {
  delete bindingAppParams;
  delete bindingNetworkParams;
  delete modelLogger;
  delete qtSimpleLogger;
  delete ui;
}

void MainWindow::onActionLoadNeuralNetwork() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      "JSON (*.json)");

  if (fileName.isEmpty()) {
    return; // No file selected
  }

  ui->lineEditCurrentNetwork->setText("");

  QFile file(fileName);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
    QMessageBox::warning(this, tr("Error"), tr("Cannot open file"));
    return;
  }

  ui->lineEditCurrentNetwork->setText(fileName);
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

  progressDialog->setValue(0);
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
  SimpleLogger::LOG_INFO(tr("Loading canceled").toStdString());
  statusBar()->showMessage(tr("Loading canceled"),
                           5000); // Show message for 5 seconds
}

void MainWindow::onLoadingFinished() {
  if (progressDialog) {
    progressDialog->setValue(100);
    progressDialog->close();
    progressDialog->deleteLater();
  }
  SimpleLogger::LOG_INFO(tr("Loading finished").toStdString());
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
        SimpleLogger::LOG_ERROR(message.toStdString());
        statusBar()->showMessage(tr("Error: %1").arg(message),
                                 5000); // Show message for 5 seconds
        QMessageBox::warning(this, tr("Error"), message);
      },
      Qt::QueuedConnection);
}

void MainWindow::onActionSelectInputFile() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select an input file as a valid image..."), "",
      tr("Image Files (*.bmp *.jpg *.jpeg *.png)"));

  if (fileName.isEmpty()) {
    return; // No file selected
  }

  ui->lineEditInputFile->setText(fileName);
}

void MainWindow::onActionSelectOutputFile() {
  auto fileName = QFileDialog::getSaveFileName(
      this,
      tr("Select or enter an output file name for the generated image..."), "",
      tr("Image Files (*.bmp *.jpg *.jpeg *.png)"));

  if (fileName.isEmpty()) {
    return; // No file selected
  }

  ui->lineEditOutputFile->setText(fileName);
}

void MainWindow::onActionSelectTrainingFile() {
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a sipai training csv file..."), "", "CSV (*.csv)");

  if (fileName.isEmpty()) {
    return; // No file selected
  }

  ui->lineEditTrainingFile->setText(fileName);
}

void MainWindow::onActionSelectTrainingFolder() {
  auto folderName = QFileDialog::getExistingDirectory(
      this, tr("Select a sipai training folder..."), "");

  if (folderName.isEmpty()) {
    return; // No file selected
  }

  ui->lineEditTrainingFolder->setText(folderName);
}

void MainWindow::onActionAbout() {
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}

void MainWindow::loadNetwork() {
  auto &manager = Manager::getInstance();

  try {
    manager.createOrImportNetwork([this](int i) {
      QMetaObject::invokeMethod(futureWatcher, "progressValueChanged",
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
