
#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "MainWindow.h"
#include "RunnerTrainingVisitor.h"
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
      runWatcher(new QFutureWatcher<void>(this)),
      bindingAppParams(new BindingAppParams()),
      bindingNetworkParams(new BindingNetworkParams()),
      logCallback(new SimpleLoggerCallback(modelLogger))
{

  auto &manager = Manager::getInstance();

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

  // Run/Stop toolbar buttons
  connect(ui->actionRun, &QAction::triggered, this, &MainWindow::onActionRun);
  connect(ui->actionStop, &QAction::triggered, this, &MainWindow::onActionStop);
  connect(runWatcher, &QFutureWatcher<void>::finished, this,
          &MainWindow::onRunFinished);

  // Settings tab buttons
  connect(ui->pushButtonSettingsApply, &QPushButton::clicked, this,
          &MainWindow::onSettingsApply);
  connect(ui->pushButtonSettingsCancel, &QPushButton::clicked, this,
          &MainWindow::onSettingsCancel);

  // Model tab buttons
  connect(ui->pushButton_NetworkLoad, &QPushButton::clicked, this,
          &MainWindow::onModelLoad);
  connect(ui->pushButton_NetworkSave, &QPushButton::clicked, this,
          &MainWindow::onModelSave);
  connect(ui->pushButton_NetworkSaveAs, &QPushButton::clicked, this,
          &MainWindow::onModelSaveAs);
  connect(ui->pushButton_Build, &QPushButton::clicked, this,
          &MainWindow::onModelBuild);
  connect(ui->pushButton_NetworkClear, &QPushButton::clicked, this,
          &MainWindow::onModelClear);

  // Model menu actions -> same as tab buttons
  connect(ui->actionModelLoad, &QAction::triggered, this,
          &MainWindow::onModelLoad);
  connect(ui->actionModelSave, &QAction::triggered, this,
          &MainWindow::onModelSave);
  connect(ui->actionModelSave_as, &QAction::triggered, this,
          &MainWindow::onModelSaveAs);
  connect(ui->actionModelBuild, &QAction::triggered, this,
          &MainWindow::onModelBuild);
  connect(ui->actionModelClear, &QAction::triggered, this,
          &MainWindow::onModelClear);

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
                               const std::string &message)
                        { logCallback->log(timestamp, level, message); });

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

MainWindow::~MainWindow()
{
  delete bindingAppParams;
  delete bindingNetworkParams;
  delete modelLogger;
  delete logCallback;
  delete ui;
}

// --- Run / Stop ---

void MainWindow::onActionRun()
{
  auto &manager = Manager::getInstance();
  if (!manager.network) {
    QMessageBox::warning(this, tr("Error"),
                         tr("No neural network loaded. Please build or load a "
                            "model first."));
    return;
  }

  setRunningState(true);
  statusBar()->showMessage(tr("Running..."));

  QFuture<void> future = QtConcurrent::run([this]() {
    try {
      Manager::getInstance().run();
    } catch (const std::exception &ex) {
      QMetaObject::invokeMethod(this, "onErrorOccurred",
                                Q_ARG(QString, QString::fromStdString(ex.what())));
    }
  });
  runWatcher->setFuture(future);
}

void MainWindow::onActionStop()
{
  // Use the same signal mechanism as the CLI (SIGINT simulation)
  if (!stopTraining) {
    stopTraining = true;
    SimpleLogger::LOG_INFO("Stop requested. Finishing current epoch...");
    statusBar()->showMessage(tr("Stopping after current epoch..."));
  } else {
    stopTrainingNow = true;
    SimpleLogger::LOG_INFO("Force stop requested.");
    statusBar()->showMessage(tr("Force stopping..."));
  }
}

void MainWindow::onRunFinished()
{
  setRunningState(false);
  SimpleLogger::LOG_INFO("Run finished.");
  statusBar()->showMessage(tr("Run finished"), 5000);
}

void MainWindow::setRunningState(bool running)
{
  isRunning_ = running;
  ui->actionRun->setEnabled(!running);
  ui->actionStop->setEnabled(running);
}

// --- Settings tab ---

void MainWindow::onSettingsApply()
{
  // Values are already written to app_params/network_params via bindings
  SimpleLogger::LOG_INFO("Settings applied.");
  statusBar()->showMessage(tr("Settings applied"), 3000);
}

void MainWindow::onSettingsCancel()
{
  // Reload UI from current params (discards unsaved widget changes)
  bindingAppParams->reload();
  bindingNetworkParams->reload();
  SimpleLogger::LOG_INFO("Settings reverted.");
  statusBar()->showMessage(tr("Settings reverted"), 3000);
}

// --- Model tab ---

void MainWindow::onModelLoad()
{
  onActionLoadNeuralNetwork();
}

void MainWindow::onModelSave()
{
  auto &manager = Manager::getInstance();
  if (!manager.network) {
    QMessageBox::warning(this, tr("Error"), tr("No neural network to save."));
    return;
  }
  if (manager.app_params.network_to_export.empty()) {
    // Fall through to Save As if no export path set
    onModelSaveAs();
    return;
  }

  statusBar()->showMessage(tr("Saving neural network..."));
  QFuture<void> future = QtConcurrent::run([this]() {
    try {
      Manager::getInstance().exportNetwork([this](int i) {
        QMetaObject::invokeMethod(this, [this, i]() {
          statusBar()->showMessage(tr("Saving... %1%").arg(i));
        }, Qt::QueuedConnection);
      });
    } catch (const std::exception &ex) {
      QMetaObject::invokeMethod(this, "onErrorOccurred",
                                Q_ARG(QString, QString::fromStdString(ex.what())));
      return;
    }
    QMetaObject::invokeMethod(this, [this]() {
      SimpleLogger::LOG_INFO("Neural network saved.");
      statusBar()->showMessage(tr("Neural network saved"), 5000);
    }, Qt::QueuedConnection);
  });
}

void MainWindow::onModelSaveAs()
{
  auto &manager = Manager::getInstance();
  if (!manager.network) {
    QMessageBox::warning(this, tr("Error"), tr("No neural network to save."));
    return;
  }

  auto fileName = QFileDialog::getSaveFileName(
      this, tr("Save neural network as..."), "", "JSON (*.json)");
  if (fileName.isEmpty()) {
    return;
  }

  manager.app_params.network_to_export = fileName.toStdString();
  onModelSave();
}

void MainWindow::onModelBuild()
{
  progressDialog =
      new QProgressDialog("Building neural network...", "Abort", 0, 100, this);
  progressDialog->setWindowModality(Qt::WindowModal);
  connect(progressDialog, &QProgressDialog::canceled, this,
          &MainWindow::onLoadingCanceled);

  statusBar()->showMessage(tr("Building neural network..."));

  QFuture<void> future = QtConcurrent::run([this]() {
    try {
      Manager::getInstance().createOrImportNetwork([this](int i) {
        QMetaObject::invokeMethod(futureWatcher, "progressValueChanged",
                                  Q_ARG(int, i));
        if (futureWatcher->isCanceled()) {
          throw std::runtime_error("Building canceled");
        }
      });
    } catch (const std::exception &ex) {
      QMetaObject::invokeMethod(this, "onErrorOccurred",
                                Q_ARG(QString, QString::fromStdString(ex.what())));
    }
  });
  futureWatcher->setFuture(future);
  progressDialog->setValue(0);
  progressDialog->show();
}

void MainWindow::onModelClear()
{
  auto &manager = Manager::getInstance();
  if (manager.network) {
    auto reply = QMessageBox::question(
        this, tr("Clear Neural Network"),
        tr("Are you sure you want to clear the current neural network?"),
        QMessageBox::Yes | QMessageBox::No);
    if (reply == QMessageBox::No) {
      return;
    }
    manager.network.reset();
    SimpleLogger::LOG_INFO("Neural network cleared.");
    statusBar()->showMessage(tr("Neural network cleared"), 5000);
  }
}

// --- File dialogs ---

void MainWindow::onActionLoadNeuralNetwork()
{
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a Sipai neural network model Json file..."), "",
      "JSON (*.json)");

  if (fileName.isEmpty())
  {
    return; // No file selected
  }

  QFile file(fileName);
  if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
  {
    QMessageBox::warning(this, tr("Error"), tr("Cannot open file"));
    return;
  }

  Manager::getInstance().app_params.network_to_import = fileName.toStdString();

  progressDialog =
      new QProgressDialog("Loading neural network...", "Abort", 0, 100, this);
  progressDialog->setWindowModality(Qt::WindowModal);

  connect(progressDialog, &QProgressDialog::canceled, this,
          &MainWindow::onLoadingCanceled);

  // Update status bar
  statusBar()->showMessage(tr("Loading neural network..."));

  // Start the concurrent loading process
  QFuture<void> future = QtConcurrent::run([this]()
                                           { loadNetwork(); });
  futureWatcher->setFuture(future);

  progressDialog->setValue(0);
  progressDialog->show();
}

void MainWindow::onProgressUpdated(int value)
{
  if (progressDialog)
  {
    progressDialog->setValue(value);
  }
}

void MainWindow::onLoadingCanceled()
{
  if (futureWatcher->isRunning())
  {
    futureWatcher->cancel();
  }
  if (progressDialog)
  {
    progressDialog->close();
    progressDialog->deleteLater();
  }
  SimpleLogger::LOG_INFO(tr("Loading canceled").toStdString());
  statusBar()->showMessage(tr("Loading canceled"),
                           5000); // Show message for 5 seconds
}

void MainWindow::onLoadingFinished()
{
  if (progressDialog)
  {
    progressDialog->setValue(100);
    progressDialog->close();
    progressDialog->deleteLater();
  }
  SimpleLogger::LOG_INFO(tr("Loading finished").toStdString());
  statusBar()->showMessage(tr("Loading finished"),
                           5000); // Show message for 5 seconds
}

void MainWindow::onErrorOccurred(const QString &message)
{
  QMetaObject::invokeMethod(
      this,
      [this, message]()
      {
        if (progressDialog)
        {
          progressDialog->close();
          progressDialog->deleteLater();
        }
        setRunningState(false);
        SimpleLogger::LOG_ERROR(message.toStdString());
        statusBar()->showMessage(tr("Error: %1").arg(message),
                                 5000); // Show message for 5 seconds
        QMessageBox::warning(this, tr("Error"), message);
      },
      Qt::QueuedConnection);
}

void MainWindow::onActionSelectInputFile()
{
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select an input file as a valid image..."), "",
      tr("Image Files (*.bmp *.jpg *.jpeg *.png)"));

  if (fileName.isEmpty())
  {
    return; // No file selected
  }

  ui->lineEditInputFile->setText(fileName);
}

void MainWindow::onActionSelectOutputFile()
{
  auto fileName = QFileDialog::getSaveFileName(
      this,
      tr("Select or enter an output file name for the generated image..."), "",
      tr("Image Files (*.bmp *.jpg *.jpeg *.png)"));

  if (fileName.isEmpty())
  {
    return; // No file selected
  }

  ui->lineEditOutputFile->setText(fileName);
}

void MainWindow::onActionSelectTrainingFile()
{
  auto fileName = QFileDialog::getOpenFileName(
      this, tr("Select a sipai training csv file..."), "", "CSV (*.csv)");

  if (fileName.isEmpty())
  {
    return; // No file selected
  }

  ui->lineEditTrainingFile->setText(fileName);
}

void MainWindow::onActionSelectTrainingFolder()
{
  auto folderName = QFileDialog::getExistingDirectory(
      this, tr("Select a sipai training folder..."), "");

  if (folderName.isEmpty())
  {
    return; // No file selected
  }

  ui->lineEditTrainingFolder->setText(folderName);
}

void MainWindow::onActionAbout()
{
  QMessageBox::about(this, tr("About SIPAI"), aboutStr_.c_str());
}

void MainWindow::loadNetwork()
{
  auto &manager = Manager::getInstance();

  try
  {
    manager.createOrImportNetwork([this](int i)
                                  {
      QMetaObject::invokeMethod(futureWatcher, "progressValueChanged",
                                Q_ARG(int, i));
      // Check for cancellation
      if (futureWatcher->isCanceled()) {
        throw std::runtime_error("Loading canceled");
      } });
  }
  catch (const std::exception &ex)
  {
    QMetaObject::invokeMethod(this, "onErrorOccurred",
                              Q_ARG(QString, ex.what()));
  }
}
