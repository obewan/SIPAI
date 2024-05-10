#include "Layer.h"
#include "Manager.h"
#include "RunnerTrainingVulkanVisitor.h"
#include "VulkanController.h"
#include "doctest.h"
#include <opencv2/core/matx.hpp>

using namespace sipai;

// Skip this test on Github, no vulkan device there.
TEST_CASE("Testing VulkanController" * doctest::skip(true)) {
  auto &manager = Manager::getInstance();
  auto &app_params = manager.app_params;
  app_params.forwardShader = "../../" + app_params.forwardShader;
  app_params.backwardShader = "../../" + app_params.backwardShader;
  app_params.training_data_file = "";
  app_params.training_data_folder = "../data/images/target/";
  app_params.max_epochs = 2;
  app_params.run_mode = ERunMode::TrainingMonitored;
  app_params.network_to_export = "";
  app_params.network_to_import = "";
  app_params.enable_vulkan = true;
  app_params.random_loading = true;
  auto &np = manager.network_params;
  np.input_size_x = 2;
  np.input_size_y = 2;
  np.hidden_size_x = 3;
  np.hidden_size_y = 2;
  np.output_size_x = 3;
  np.output_size_y = 3;
  np.hiddens_count = 1;
  np.learning_rate = 0.5f;
  np.adaptive_learning_rate = true;
  np.adaptive_learning_rate_factor = 0.123f;
  manager.createOrImportNetwork();

  CHECK(sizeof(cv::Vec4f) == 4 * sizeof(float));

  auto &vulkanController = VulkanController::getInstance();

  CHECK_NOTHROW(vulkanController.destroy());
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() == VK_NULL_HANDLE);

  CHECK_NOTHROW(vulkanController.initialize());
  CHECK(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() != VK_NULL_HANDLE);

  // test forward propagation from previous to current layer
  auto currentLayer = manager.network->layers.back();
  CHECK(currentLayer->layerType == LayerType::LayerOutput);
  auto previousLayer = currentLayer->previousLayer;
  CHECK(previousLayer->layerType == LayerType::LayerHidden);

  auto oldValues = currentLayer->values.clone();
  RunnerTrainingVulkanVisitor visitor;
  CHECK_NOTHROW(visitor.visit());
  auto newValues = currentLayer->values.clone();

  std::cout << "values before:\n" << oldValues << std::endl;
  std::cout << "values after:\n" << newValues << std::endl;

  //  Check if oldValues and newValues are not equals
  cv::Mat diff = cv::abs(oldValues - newValues);
  double norm = cv::norm(diff, cv::NORM_L1);
  bool isEq = norm < std::numeric_limits<float>::epsilon() * diff.total();
  CHECK(isEq == false);

  vulkanController.destroy();
  CHECK_FALSE(vulkanController.IsInitialized());
  CHECK(vulkanController.getDevice() == VK_NULL_HANDLE);

  manager.network.reset();
}