#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)
#include "NetworkLoader.h"

using namespace Qt::StringLiterals;
using namespace sipai;

NetworkLoader::NetworkLoader(QObject *parent) : QObject(parent) {}

void NetworkLoader::setFileName(const QString &fileName) {
  m_fileName = fileName;
}

void NetworkLoader::loadNetwork() {
  auto &manager = Manager::getInstance();
  manager.app_params.network_to_import = m_fileName.toStdString();

  try {
    manager.createOrImportNetwork([this](int i) {
      emit progressUpdated(i);
      // Here we can check for cancellation if needed
    });
    emit loadingFinished();
  } catch (const std::exception &ex) {
    emit errorOccurred(ex.what());
  }
}