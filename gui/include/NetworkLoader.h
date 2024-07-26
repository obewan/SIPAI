/**
 * @file NetworkLoader.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief Network loader class for the ui
 * @date 2024-07-27
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */

#include <QObject>
#include <functional>

class NetworkLoader : public QObject {
  Q_OBJECT

public:
  explicit NetworkLoader(QObject *parent = nullptr);
  void setFileName(const QString &fileName);

signals:
  void progressUpdated(int value);
  void loadingFinished();
  void errorOccurred(const QString &message);

public slots:
  void loadNetwork();

private:
  QString m_fileName;
};
