#include "doctest.h"
#include "exception/EmptyCellException.h"
#include "exception/FileReaderException.h"
#include "exception/ImageHelperException.h"
#include "exception/ImportExportException.h"
#include "exception/ManagerException.h"
#include "exception/NetworkException.h"

using namespace sipai;

TEST_CASE("Testing exceptions") {
  SUBCASE("Test constructors") {
    CHECK_THROWS_AS_MESSAGE({ throw EmptyCellException("test"); },
                            EmptyCellException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw FileReaderException("test"); },
                            FileReaderException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw ImageHelperException("test"); },
                            ImageHelperException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw ImportExportException("test"); },
                            ImportExportException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw ManagerException("test"); },
                            ManagerException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw NetworkException("test"); },
                            NetworkException, "test");
  }
}