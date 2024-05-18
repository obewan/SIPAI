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

  SUBCASE("Test template build") {
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

  SUBCASE("Test getBuffer") {
    auto &controller = VulkanController::getInstance();
    auto vulkan = controller.getVulkan();
    vulkan->buffers.push_back(Buffer{.name = EBuffer::InputData, .binding = 1});
    vulkan->buffers.push_back(
        Buffer{.name = EBuffer::OutputData, .binding = 2});

    auto &buf1 = controller.getBuffer(EBuffer::InputData);
    CHECK(buf1.name == EBuffer::InputData);
    CHECK(buf1.binding == 1);

    auto &buf2 = controller.getBuffer(EBuffer::OutputData);
    CHECK(buf2.name == EBuffer::OutputData);
    CHECK(buf2.binding == 2);

    buf1.binding = 3;
    auto &buf1b = controller.getBuffer(EBuffer::InputData);
    CHECK(buf1b.name == EBuffer::InputData);
    CHECK(buf1b.binding == 3);

    CHECK_THROWS_AS(controller.getBuffer(EBuffer::OutputLayer),
                    VulkanControllerException);
  }

  SUBCASE("Test various") {
    CHECK(sizeof(GLSLNeuron) ==
          (2 * sizeof(uint) + MAX_NEIGHBORS * sizeof(GLSLNeighbor) +
           sizeof(std::vector<std::vector<cv::Vec4f>>)));
  }
}