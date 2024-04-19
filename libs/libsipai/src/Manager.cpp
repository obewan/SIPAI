#include "Manager.h"
#include "AppParams.h"
#include "Common.h"
#include "NeuralNetwork.h"
#include "NeuralNetworkBuilder.h"
#include "SimpleLogger.h"
#include "TrainingDataFileReaderCSV.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <opencv2/core/types.hpp>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>

using namespace sipai;

std::unique_ptr<Manager> Manager::instance_ = nullptr;

void Manager::initializeVulkan() {
  // Initialize Vulkan
  app_params.vulkan = false;
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "SIPAI";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo createInfoInstance{};
  createInfoInstance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfoInstance.pApplicationInfo = &appInfo;

  if (vkCreateInstance(&createInfoInstance, nullptr, &vkInstance_) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create instance!");
  }

  // Create a device
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, nullptr);

  if (deviceCount == 0) {
    throw std::runtime_error("failed to find GPUs with Vulkan support!");
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(vkInstance_, &deviceCount, devices.data());

  for (const auto &device : devices) {
    if (isDeviceSuitable(device)) {
      vkPhysicalDevice_ = device;
      break;
    }
  }

  if (vkPhysicalDevice_ == VK_NULL_HANDLE) {
    throw std::runtime_error("failed to find a suitable GPU!");
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &queueFamilyCount,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_, &queueFamilyCount,
                                           queueFamilies.data());

  uint32_t queueFamilyIndex = 0;
  int i = 0;
  for (const auto &queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      queueFamilyIndex = i;
      break;
    }
    i++;
  }

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.logicOp = VK_TRUE; // Enable logical operation feature
  deviceFeatures.shaderFloat64 =
      VK_TRUE; // Enable 64-bit floats in shader code feature

  VkDeviceCreateInfo createInfoDevice{};
  createInfoDevice.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfoDevice.pQueueCreateInfos = &queueCreateInfo;
  createInfoDevice.queueCreateInfoCount = 1;
  createInfoDevice.pEnabledFeatures = &deviceFeatures;

  if (vkCreateDevice(vkPhysicalDevice_, &createInfoDevice, nullptr,
                     &vkDevice_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create logical device!");
  }
  app_params.vulkan = true;
}

void Manager::createOrImportNetwork() {
  if (!network) {
    auto builder = std::make_unique<NeuralNetworkBuilder>();
    network = builder->createOrImport()
                  .addLayers()
                  .bindLayers()
                  .addNeighbors()
                  .initializeWeights()
                  .setActivationFunction()
                  .build();
  }
}

void Manager::exportNetwork() {
  if (!app_params.network_to_export.empty()) {
    SimpleLogger::LOG_INFO("Saving the neural network to ",
                           app_params.network_to_export, " and ",
                           getFilenameCsv(app_params.network_to_export), "...");
    auto exportator = std::make_unique<NeuralNetworkImportExportFacade>();
    exportator->exportModel(network, network_params, app_params);
  }
}

void Manager::run() {
  SimpleLogger::LOG_INFO(getVersionHeader());

  // Initialize network
  createOrImportNetwork();

  SimpleLogger::LOG_INFO(
      "Parameters: ", "\nmode: ", getRunModeStr(app_params.run_mode),
      "\nauto-save every ", app_params.epoch_autosave, " epochs",
      "\nauto-exit after ", app_params.max_epochs_without_improvement,
      " epochs without improvement",
      app_params.max_epochs == NO_MAX_EPOCHS
          ? "\nno maximum epochs"
          : "\nauto-exit after a maximum of " +
                std::to_string(app_params.max_epochs) + " epochs",
      "\ntraining/validation ratio: ", app_params.training_split_ratio,
      "\nlearning rate: ", network_params.learning_rate,
      "\nadaptive learning rate: ",
      network_params.adaptive_learning_rate ? "true" : "false",
      "\nadaptive learning rate increase: ",
      network_params.enable_adaptive_increase ? "true" : "false",
      "\nadaptive learning rate factor: ",
      network_params.adaptive_learning_rate_factor,
      "\ntraining error min: ", network_params.error_min,
      "\ntraining error max: ", network_params.error_max,
      "\ninput layer size: ", network_params.input_size_x, "x",
      network_params.input_size_y,
      "\nhidden layer size: ", network_params.hidden_size_x, "x",
      network_params.hidden_size_y,
      "\noutput layer size: ", network_params.output_size_x, "x",
      network_params.output_size_y,
      "\nhidden layers: ", network_params.hiddens_count,
      "\nhidden activation function: ",
      getActivationStr(network_params.hidden_activation_function),
      "\nhidden activation alpha: ", network_params.hidden_activation_alpha,
      "\noutput activation function: ",
      getActivationStr(network_params.output_activation_function),
      "\noutput activation alpha: ", network_params.output_activation_alpha,
      "\nimage split: ", app_params.image_split,
      "\ninput reduce factor: ", app_params.training_reduce_factor,
      "\noutput scale: ", app_params.output_scale,
      "\nimages random loading: ", app_params.random_loading ? "true" : "false",
      "\nimages bulk loading: ", app_params.bulk_loading ? "true" : "false",
      "\npadding enabled: ", app_params.enable_padding ? "true" : "false",
      "\nparallelism enabled: ", app_params.enable_parallel ? "true" : "false",
      "\nverbose logs enabled: ", app_params.verbose ? "true" : "false");

  // Run with visitor
  switch (app_params.run_mode) {
  case ERunMode::TrainingMonitored:
    runWithVisitor(runnerVisitorFactory_.getTrainingMonitoredVisitor());
    break;
  case ERunMode::Enhancer:
    runWithVisitor(runnerVisitorFactory_.getEnhancerVisitor());
    break;
  default:
    break;
  }
}

void Manager::runWithVisitor(const RunnerVisitor &visitor) { visitor.visit(); }

bool Manager::isDeviceSuitable(const VkPhysicalDevice &device) {
  VkPhysicalDeviceProperties deviceProperties;
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  // This feature indicates whether the physical device supports logical
  // operations in the output color blending. Logical operations are used to
  // combine color and/or alpha components of the source and destination
  // fragments in ways other than traditional blending+
  bool logicOpShaderSupported = deviceFeatures.logicOp;

  // This feature indicates whether the physical device supports 64-bit floats
  // (doubles) in shader code. If this feature is not enabled, you won’t be
  // able to use 64-bit floats in your shaders.
  bool anisotropicFilteringSupported = deviceFeatures.shaderFloat64;

  return logicOpShaderSupported && anisotropicFilteringSupported;
}

void Manager::clearVulkan() {
  vkDestroyDevice(vkDevice_, nullptr);
  vkDestroyInstance(vkInstance_, nullptr);
}