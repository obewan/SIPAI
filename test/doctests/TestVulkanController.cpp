#include "Layer.h"
#include "Manager.h"
#include "RunnerTrainingVulkanVisitor.h"
#include "VulkanController.h"
#include "doctest.h"
#include <filesystem>
#include <fstream>

using namespace sipai;

// Skip this test on Github, no vulkan device there.
TEST_CASE("Testing VulkanController" * doctest::skip(true)) {

  SUBCASE("Test template builder") {
    const auto &manager = Manager::getConstInstance();
    VulkanHelper helper;
    std::string relativePath = "../../";

    CHECK(std::filesystem::exists(
        relativePath + manager.app_params.trainingMonitoredShaderTemplate));

    if (std::filesystem::exists(relativePath +
                                manager.app_params.trainingMonitoredShader)) {
      std::filesystem::remove(relativePath +
                              manager.app_params.trainingMonitoredShader);
    }
    CHECK(helper.replaceTemplateParameters(
        relativePath + manager.app_params.trainingMonitoredShaderTemplate,
        relativePath + manager.app_params.trainingMonitoredShader));

    CHECK(std::filesystem::exists(relativePath +
                                  manager.app_params.trainingMonitoredShader));

    std::ifstream inFile(relativePath +
                         manager.app_params.trainingMonitoredShader);
    std::string line;
    while (std::getline(inFile, line)) {
      CHECK(line.find("%%") == std::string::npos);
    }
  }
}