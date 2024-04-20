#include "Manager.h"
#include "VulkanController.h"
#include "doctest.h"

using namespace sipai;

// Ignore this test on Github, no vulkan device there.
TEST_CASE("Testing VulkanController" * doctest::skip(true)) {
  auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;
  app_params.forwardShader = "../../" + app_params.forwardShader;

  auto &vulkanController = VulkanController::getInstance();

  CHECK_NOTHROW(vulkanController.destroy());
  CHECK_FALSE(vulkanController.IsInitialized());

  CHECK_NOTHROW(vulkanController.initialize());
  CHECK(vulkanController.IsInitialized());
  CHECK(vulkanController.getVkInstance() != VK_NULL_HANDLE);
  CHECK(vulkanController.getVkDevice() != VK_NULL_HANDLE);
  CHECK(vulkanController.getVkPhysicalDevice() != VK_NULL_HANDLE);

  vulkanController.destroy();
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getVkInstance() == VK_NULL_HANDLE);
  CHECK(vulkanController.getVkDevice() == VK_NULL_HANDLE);
  CHECK(vulkanController.getVkPhysicalDevice() == VK_NULL_HANDLE);
}