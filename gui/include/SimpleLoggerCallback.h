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
#include <QStandardItemModel>

class SimpleLoggerCallback {
public:
  SimpleLoggerCallback(QStandardItemModel *model) : modelLogger(model) {}

  void log(const std::string &timestamp, const std::string &level,
           const std::string &message) {
    QList<QStandardItem *> items;
    items.append(new QStandardItem(QString::fromStdString(timestamp)));
    items.append(new QStandardItem(QString::fromStdString(level)));
    items.append(new QStandardItem(QString::fromStdString(message)));
    modelLogger->appendRow(items);
  }

private:
  QStandardItemModel *modelLogger;
};