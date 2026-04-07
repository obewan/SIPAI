/**
 * @file SimpleLoggerCallback.h
 * @author Damien Balima (www.dams-labs.net)
 * @brief A Log Callback for sipai::SimpleLogger to use with Qt.
 * @date 2024-08-05
 *
 * @copyright Damien Balima (c) CC-BY-NC-SA-4.0 2024
 *
 */
#pragma once
#include <QList>
#include <QMetaObject>
#include <QStandardItemModel>

class SimpleLoggerCallback : public QObject {
  Q_OBJECT
public:
  SimpleLoggerCallback(QStandardItemModel *model, QObject *parent = nullptr)
      : QObject(parent), modelLogger(model) {}

  void log(const std::string &timestamp, const std::string &level,
           const std::string &message) {
    QString ts = QString::fromStdString(timestamp);
    QString lv = QString::fromStdString(level);
    QString msg = QString::fromStdString(message);
    // Marshal to the UI thread for thread-safe model access
    QMetaObject::invokeMethod(
        this, [this, ts, lv, msg]() { appendRow(ts, lv, msg); },
        Qt::QueuedConnection);
  }

private:
  void appendRow(const QString &timestamp, const QString &level,
                 const QString &message) {
    QList<QStandardItem *> items;
    items.append(new QStandardItem(timestamp));
    items.append(new QStandardItem(level));
    items.append(new QStandardItem(message));
    modelLogger->appendRow(items);
  }

  QStandardItemModel *modelLogger;
};
