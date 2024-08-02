#include "Manager.h"
#undef emit // Undefine the TBB emit macro to avoid conflicts (workaround)

#include "./ui_MainWindow.h"
#include "Bindings.h"
#include "MainWindow.h"

using namespace Qt::StringLiterals;
using namespace sipai;

void Bindings::setBindings(Ui::MainWindow* ui){
  connect(ui->lineEditCurrentNetwork, &QLineEdit::textChanged, this,
          [this](const QString &text) {
            setNetworkToImport(text.toStdString());
          });
}


void Bindings::setNetworkToImport(const std::string &value) {
  Manager::getInstance().app_params.network_to_import = value;
}