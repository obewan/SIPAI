#include "doctest.h"
#include "exception/EmptyCellException.h"
#include "exception/FileReaderException.h"
#include "exception/ImageHelperException.h"
#include "exception/ImportExportException.h"
#include "exception/ManagerException.h"
#include "exception/NeuralNetworkException.h"
#include "exception/TrainingDataFactoryException.h"
#include "exception/VulkanBuilderException.h"
#include "exception/VulkanControllerException.h"

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
    CHECK_THROWS_AS_MESSAGE({ throw NeuralNetworkException("test"); },
                            NeuralNetworkException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw TrainingDataFactoryException("test"); },
                            TrainingDataFactoryException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw VulkanControllerException("test"); },
                            VulkanControllerException, "test");
    CHECK_THROWS_AS_MESSAGE({ throw VulkanBuilderException("test"); },
                            VulkanBuilderException, "test");
  }
}