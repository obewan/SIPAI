#pragma once
#include "ImportExportException.h"

namespace sipai {
/**
 * @brief EmptyCellException
 *
 */
class EmptyCellException : public ImportExportException {
public:
  using ImportExportException::ImportExportException;
};

} // namespace sipai