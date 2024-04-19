#include "Manager.h"
#include "VulkanController.h"
#include "doctest.h"

using namespace sipai;

// Ignore this test on Github, no vulkan device there.
TEST_CASE("Testing VulkanController" * doctest::skip(true)) {
  auto &vulkanController = VulkanController::getInstance();
  CHECK_NOTHROW(vulkanController.destroy());
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK_FALSE(Manager::getInstance().app_params.vulkan);

  CHECK_NOTHROW(vulkanController.initialize());
  CHECK(vulkanController.IsInitialized());
  CHECK(vulkanController.getVkInstance() != VK_NULL_HANDLE);
  CHECK(vulkanController.getVkDevice() != VK_NULL_HANDLE);
  CHECK(vulkanController.getVkPhysicalDevice() != VK_NULL_HANDLE);
  CHECK(Manager::getInstance().app_params.vulkan);

  vulkanController.destroy();
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getVkInstance() == VK_NULL_HANDLE);
  CHECK(vulkanController.getVkDevice() == VK_NULL_HANDLE);
  CHECK(vulkanController.getVkPhysicalDevice() == VK_NULL_HANDLE);
  CHECK_FALSE(Manager::getInstance().app_params.vulkan);
}