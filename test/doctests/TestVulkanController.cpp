#include "Manager.h"
#include "VulkanController.h"
#include "doctest.h"

using namespace sipai;

// Skip this test on Github, no vulkan device there.
TEST_CASE("Testing VulkanController" * doctest::skip(true)) {
  auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;
  app_params.forwardShader = "../../" + app_params.forwardShader;
  app_params.network_to_import = "";
  auto &np = manager.network_params;
  np.input_size_x = 2;
  np.input_size_y = 2;
  np.hidden_size_x = 3;
  np.hidden_size_y = 2;
  np.output_size_x = 3;
  np.output_size_y = 3;
  np.hiddens_count = 1;
  np.learning_rate = 0.02f;
  np.adaptive_learning_rate = true;
  np.adaptive_learning_rate_factor = 0.123;
  manager.createOrImportNetwork();

  auto &vulkanController = VulkanController::getInstance();

  CHECK_NOTHROW(vulkanController.destroy());
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() == VK_NULL_HANDLE);

  CHECK_NOTHROW(vulkanController.initialize());
  CHECK(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() != VK_NULL_HANDLE);
  
  vulkanController.destroy();
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() == VK_NULL_HANDLE);
  
  manager.network.reset();
}