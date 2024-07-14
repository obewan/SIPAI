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
    CHECK(std::filesystem::exists(relativePath +
                                  manager.app_params.shaderTrainingTemplate));
    if (std::filesystem::exists(relativePath +
                                manager.app_params.shaderTraining)) {
      std::filesystem::remove(relativePath + manager.app_params.shaderTraining);
    }
    CHECK(helper.replaceTemplateParameters(
        relativePath + manager.app_params.shaderTrainingTemplate,
        relativePath + manager.app_params.shaderTraining));
    CHECK(std::filesystem::exists(relativePath +
                                  manager.app_params.shaderTraining));
    std::ifstream inFile(relativePath + manager.app_params.shaderTraining);
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

  SUBCASE("Test Vulkan basic") {
    auto &manager = Manager::getInstance();
    manager.network.reset();
    auto &ap = manager.app_params;
    auto &np = manager.network_params;
    np.input_size_x = 10;
    np.input_size_y = 10;
    np.hidden_size_x = 10;
    np.hidden_size_y = 10;
    np.output_size_x = 10;
    np.output_size_y = 10;
    np.hiddens_count = 1;
    np.learning_rate = 0.65f;
    np.adaptive_learning_rate = true;
    np.adaptive_learning_rate_factor = 0.123f;
    ap.network_to_import = "";
    ap.network_to_export = "";
    CHECK_NOTHROW(manager.createOrImportNetwork());

    VulkanBuilder builder;
    auto vulkan = std::make_shared<Vulkan>();
    builder.withVulkan(vulkan).initialize();

    // Create buffer
    VkBuffer paramsBuffer;
    GLSLParameters params{
        .learning_rate = 0.65f, .error_min = 0.1f, .error_max = 0.9f};
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(GLSLParameters);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(vulkan->logicalDevice, &bufferCreateInfo, nullptr,
                   &paramsBuffer);

    // Allocate memory for buffer
    VkMemoryPropertyFlags memoryPropertiesFlags = builder.getMemoryProperties();
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan->logicalDevice, paramsBuffer,
                                  &memRequirements);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = builder.findMemoryType(
        memRequirements.memoryTypeBits, memoryPropertiesFlags);
    VkDeviceMemory bufferMemory;
    vkAllocateMemory(vulkan->logicalDevice, &allocInfo, nullptr, &bufferMemory);
    vkBindBufferMemory(vulkan->logicalDevice, paramsBuffer, bufferMemory, 0);

    // Copy data to buffer memory
    void *data;
    vkMapMemory(vulkan->logicalDevice, bufferMemory, 0, sizeof(GLSLParameters),
                0, &data);
    memcpy(data, &params, sizeof(GLSLParameters));
    vkUnmapMemory(vulkan->logicalDevice, bufferMemory);

    // Read data from buffer
    GLSLParameters readParams;
    vkMapMemory(vulkan->logicalDevice, bufferMemory, 0, sizeof(GLSLParameters),
                0, &data);
    memcpy(&readParams, data, sizeof(GLSLParameters));
    vkUnmapMemory(vulkan->logicalDevice, bufferMemory);

    CHECK(readParams.learning_rate == doctest::Approx(params.learning_rate));

    // Cleanup
    vkDestroyBuffer(vulkan->logicalDevice, paramsBuffer, nullptr);
    vkFreeMemory(vulkan->logicalDevice, bufferMemory, nullptr);

    // Destroy device and instance
    builder.clear();
  }

  SUBCASE("Test Vulkan basic with shader") {
    struct OutputLoss {
      float loss;
    };

    auto &manager = Manager::getInstance();
    manager.network.reset();
    auto &ap = manager.app_params;
    auto &np = manager.network_params;
    np.input_size_x = 10;
    np.input_size_y = 10;
    np.hidden_size_x = 10;
    np.hidden_size_y = 10;
    np.output_size_x = 10;
    np.output_size_y = 10;
    np.hiddens_count = 1;
    np.learning_rate = 0.65f;
    np.adaptive_learning_rate = true;
    np.adaptive_learning_rate_factor = 0.123f;
    ap.network_to_import = "";
    ap.network_to_export = "";
    CHECK_NOTHROW(manager.createOrImportNetwork());

    VulkanBuilder builder;
    auto vulkan = std::make_shared<Vulkan>();
    builder.withVulkan(vulkan).initialize();

    VkResult result;

    auto computeShaderCode =
        builder.loadShader("../../test/data/shaders/shader_test1.comp");

    VkMemoryPropertyFlags memoryPropertiesFlags = builder.getMemoryProperties();

    // Create buffer params
    VkBuffer paramsBuffer;
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(GLSLParameters);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(vulkan->logicalDevice, &bufferCreateInfo, nullptr,
                   &paramsBuffer);
    // Allocate memory for buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan->logicalDevice, paramsBuffer,
                                  &memRequirements);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = builder.findMemoryType(
        memRequirements.memoryTypeBits, memoryPropertiesFlags);
    VkDeviceMemory paramsBufferMemory;
    result = vkAllocateMemory(vulkan->logicalDevice, &allocInfo, nullptr,
                              &paramsBufferMemory);
    CHECK(result == VK_SUCCESS);
    result = vkBindBufferMemory(vulkan->logicalDevice, paramsBuffer,
                                paramsBufferMemory, 0);
    CHECK(result == VK_SUCCESS);

    // Create buffer loss
    VkBuffer outputLossBuffer;
    VkBufferCreateInfo bufferCreateInfo2 = {};
    bufferCreateInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo2.size = sizeof(GLSLParameters);
    bufferCreateInfo2.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(vulkan->logicalDevice, &bufferCreateInfo2, nullptr,
                   &outputLossBuffer);
    // Allocate memory for buffer
    VkMemoryRequirements memRequirements2;
    vkGetBufferMemoryRequirements(vulkan->logicalDevice, outputLossBuffer,
                                  &memRequirements2);
    VkMemoryAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo2.allocationSize = memRequirements2.size;
    allocInfo2.memoryTypeIndex = builder.findMemoryType(
        memRequirements2.memoryTypeBits, memoryPropertiesFlags);
    VkDeviceMemory outputLossBufferMemory;
    result = vkAllocateMemory(vulkan->logicalDevice, &allocInfo2, nullptr,
                              &outputLossBufferMemory);
    CHECK(result == VK_SUCCESS);
    result = vkBindBufferMemory(vulkan->logicalDevice, outputLossBuffer,
                                outputLossBufferMemory, 0);
    CHECK(result == VK_SUCCESS);

    // Create descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(2);
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(2);
    result = vkCreateDescriptorPool(vulkan->logicalDevice, &poolInfo, nullptr,
                                    &vulkan->descriptorPool);
    CHECK(result == VK_SUCCESS);

    // Create descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 2> descriptorLayouts;
    descriptorLayouts[0].binding = 0; // Matches 'binding = 0' in shader
    descriptorLayouts[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorLayouts[0].descriptorCount = 1;
    descriptorLayouts[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorLayouts[1].binding = 6; // Matches 'binding = 6' in shader
    descriptorLayouts[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorLayouts[1].descriptorCount = 1;
    descriptorLayouts[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(descriptorLayouts.size());
    layoutInfo.pBindings = descriptorLayouts.data(); // array of bindings
    result = vkCreateDescriptorSetLayout(vulkan->logicalDevice, &layoutInfo,
                                         nullptr, &vulkan->descriptorSetLayout);
    CHECK(result == VK_SUCCESS);

    // Create descriptor set using previous descriptorSetLayout
    VkDescriptorSetAllocateInfo allocInfoSet = {};
    allocInfoSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoSet.descriptorPool = vulkan->descriptorPool;
    allocInfoSet.descriptorSetCount = 1;
    allocInfoSet.pSetLayouts = &vulkan->descriptorSetLayout;
    result = vkAllocateDescriptorSets(vulkan->logicalDevice, &allocInfoSet,
                                      &vulkan->descriptorSet);
    CHECK(result == VK_SUCCESS);

    // Update descriptor sets (bind buffers)
    VkDescriptorBufferInfo paramsBufferInfo = {};
    paramsBufferInfo.buffer = paramsBuffer;
    paramsBufferInfo.offset = 0;
    paramsBufferInfo.range = sizeof(GLSLParameters);

    VkDescriptorBufferInfo outputLossBufferInfo = {};
    outputLossBufferInfo.buffer = outputLossBuffer;
    outputLossBufferInfo.offset = 0;
    outputLossBufferInfo.range = sizeof(OutputLoss);

    std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};

    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet =
        vulkan->descriptorSet; // Assume descriptorSet is already created
    descriptorWrites[0].dstBinding = 0; // Matches 'binding = 0' in shader
    descriptorWrites[0].dstArrayElement = 0;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].pBufferInfo = &paramsBufferInfo;

    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = vulkan->descriptorSet;
    descriptorWrites[1].dstBinding = 6; // Matches 'binding = 6' in shader
    descriptorWrites[1].dstArrayElement = 0;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].pBufferInfo = &outputLossBufferInfo;

    vkUpdateDescriptorSets(vulkan->logicalDevice,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);

    // Create shader module
    VkShaderModuleCreateInfo createInfoSM = {};
    createInfoSM.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfoSM.codeSize = computeShaderCode->size() * sizeof(uint32_t);
    createInfoSM.pCode = computeShaderCode->data();
    VkShaderModule computeShaderModule;
    result = vkCreateShaderModule(vulkan->logicalDevice, &createInfoSM, nullptr,
                                  &computeShaderModule);
    CHECK(result == VK_SUCCESS);

    // Create pipeline layout
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout setLayouts[] = {vulkan->descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    result = vkCreatePipelineLayout(vulkan->logicalDevice, &pipelineLayoutInfo,
                                    nullptr, &pipelineLayout);
    CHECK(result == VK_SUCCESS);

    // Create compute pipeline
    VkPipeline computePipeline;
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = computeShaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = 0;
    result = vkCreateComputePipelines(vulkan->logicalDevice, VK_NULL_HANDLE, 1,
                                      &pipelineInfo, nullptr, &computePipeline);
    CHECK(result == VK_SUCCESS);

    // Create command pool
    VkCommandPoolCreateInfo poolInfoCMD = {};
    poolInfoCMD.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfoCMD.queueFamilyIndex = vulkan->queueComputeIndex;
    poolInfoCMD.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    result = vkCreateCommandPool(vulkan->logicalDevice, &poolInfoCMD, nullptr,
                                 &vulkan->commandPool);
    CHECK(result == VK_SUCCESS);

    // Create command buffer pool
    VkCommandBufferAllocateInfo allocInfoCBP = {};
    allocInfoCBP.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfoCBP.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfoCBP.commandPool = vulkan->commandPool;
    allocInfoCBP.commandBufferCount = 1;
    vulkan->commandBufferPool = std::vector<VkCommandBuffer>(1);
    result = vkAllocateCommandBuffers(vulkan->logicalDevice, &allocInfoCBP,
                                      vulkan->commandBufferPool.data());
    CHECK(result == VK_SUCCESS);

    // Copy data to buffer memory
    GLSLParameters params{
        .learning_rate = 0.65f, .error_min = 0.1f, .error_max = 0.9f};
    void *paramsData;
    result = vkMapMemory(vulkan->logicalDevice, paramsBufferMemory, 0,
                         sizeof(GLSLParameters), 0, &paramsData);
    CHECK(result == VK_SUCCESS);
    memcpy(paramsData, &params, sizeof(GLSLParameters));
    vkUnmapMemory(vulkan->logicalDevice, paramsBufferMemory);

    // Run the shader
    VkCommandBuffer commandBuffer = vulkan->commandBufferPool.back();
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);
    VkDescriptorSet descriptorSets[] = {vulkan->descriptorSet};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, descriptorSets, 0, nullptr);
    vkCmdDispatch(commandBuffer, 1, 1, 1);
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(vulkan->queueCompute, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkan->queueCompute);

    // Read data from buffer
    OutputLoss outputLoss{.loss = 0.0f};
    void *outputData;
    result = vkMapMemory(vulkan->logicalDevice, outputLossBufferMemory, 0,
                         sizeof(OutputLoss), 0, &outputData);
    CHECK(result == VK_SUCCESS);
    outputLoss.loss = *reinterpret_cast<float *>(outputData);
    vkUnmapMemory(vulkan->logicalDevice, outputLossBufferMemory);
    CHECK(outputLoss.loss == doctest::Approx(params.learning_rate * 2));

    // Cleanup
    vkDestroyBuffer(vulkan->logicalDevice, paramsBuffer, nullptr);
    vkFreeMemory(vulkan->logicalDevice, paramsBufferMemory, nullptr);
    vkDestroyBuffer(vulkan->logicalDevice, outputLossBuffer, nullptr);
    vkFreeMemory(vulkan->logicalDevice, outputLossBufferMemory, nullptr);
    vkDestroyShaderModule(vulkan->logicalDevice, computeShaderModule, nullptr);
    vkDestroyPipeline(vulkan->logicalDevice, computePipeline, nullptr);
    vkDestroyPipelineLayout(vulkan->logicalDevice, pipelineLayout, nullptr);
    builder.clear();
  }

  SUBCASE("Test Vulkan basic with shader 2") {
    struct OutputLoss {
      float loss;
    };

    auto &manager = Manager::getInstance();
    manager.network.reset();
    auto &ap = manager.app_params;
    auto &np = manager.network_params;
    np.input_size_x = 10;
    np.input_size_y = 10;
    np.hidden_size_x = 10;
    np.hidden_size_y = 10;
    np.output_size_x = 10;
    np.output_size_y = 10;
    np.hiddens_count = 1;
    np.learning_rate = 0.65f;
    np.adaptive_learning_rate = true;
    np.adaptive_learning_rate_factor = 0.123f;
    ap.network_to_import = "";
    ap.network_to_export = "";
    CHECK_NOTHROW(manager.createOrImportNetwork());

    VulkanBuilder builder;
    auto vulkan = std::make_shared<Vulkan>();
    builder.withVulkan(vulkan).initialize();

    VkResult result;

    auto computeShaderCode =
        builder.loadShader("../../test/data/shaders/shader_test1.comp");

    VkMemoryPropertyFlags memoryPropertiesFlags = builder.getMemoryProperties();

    // Create buffer params
    VkBuffer paramsBuffer;
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(GLSLParameters);
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(vulkan->logicalDevice, &bufferCreateInfo, nullptr,
                   &paramsBuffer);
    // Allocate memory for buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vulkan->logicalDevice, paramsBuffer,
                                  &memRequirements);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = builder.findMemoryType(
        memRequirements.memoryTypeBits, memoryPropertiesFlags);
    VkDeviceMemory paramsBufferMemory;
    result = vkAllocateMemory(vulkan->logicalDevice, &allocInfo, nullptr,
                              &paramsBufferMemory);
    CHECK(result == VK_SUCCESS);
    result = vkBindBufferMemory(vulkan->logicalDevice, paramsBuffer,
                                paramsBufferMemory, 0);
    CHECK(result == VK_SUCCESS);

    // Create buffer loss
    VkBuffer outputLossBuffer;
    VkBufferCreateInfo bufferCreateInfo2 = {};
    bufferCreateInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo2.size = sizeof(GLSLParameters);
    bufferCreateInfo2.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo2.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(vulkan->logicalDevice, &bufferCreateInfo2, nullptr,
                   &outputLossBuffer);
    // Allocate memory for buffer
    VkMemoryRequirements memRequirements2;
    vkGetBufferMemoryRequirements(vulkan->logicalDevice, outputLossBuffer,
                                  &memRequirements2);
    VkMemoryAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo2.allocationSize = memRequirements2.size;
    allocInfo2.memoryTypeIndex = builder.findMemoryType(
        memRequirements2.memoryTypeBits, memoryPropertiesFlags);
    VkDeviceMemory outputLossBufferMemory;
    result = vkAllocateMemory(vulkan->logicalDevice, &allocInfo2, nullptr,
                              &outputLossBufferMemory);
    CHECK(result == VK_SUCCESS);
    result = vkBindBufferMemory(vulkan->logicalDevice, outputLossBuffer,
                                outputLossBufferMemory, 0);
    CHECK(result == VK_SUCCESS);

    // Create descriptor pool
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(2);
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(2);
    result = vkCreateDescriptorPool(vulkan->logicalDevice, &poolInfo, nullptr,
                                    &vulkan->descriptorPool);
    CHECK(result == VK_SUCCESS);

    // Create descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> descriptorLayouts;
    for (size_t i = 0; i < 2; i++) {
      VkDescriptorSetLayoutBinding layoutBinding;
      layoutBinding.binding = (i == 0 ? 0 : 6); // buffer binding
      layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      layoutBinding.descriptorCount = 1;
      layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      descriptorLayouts.push_back(layoutBinding);
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(descriptorLayouts.size());
    layoutInfo.pBindings = descriptorLayouts.data(); // array of bindings
    result = vkCreateDescriptorSetLayout(vulkan->logicalDevice, &layoutInfo,
                                         nullptr, &vulkan->descriptorSetLayout);
    CHECK(result == VK_SUCCESS);

    // Create descriptor set using previous descriptorSetLayout
    VkDescriptorSetAllocateInfo allocInfoSet = {};
    allocInfoSet.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoSet.descriptorPool = vulkan->descriptorPool;
    allocInfoSet.descriptorSetCount = 1;
    allocInfoSet.pSetLayouts = &vulkan->descriptorSetLayout;
    result = vkAllocateDescriptorSets(vulkan->logicalDevice, &allocInfoSet,
                                      &vulkan->descriptorSet);
    CHECK(result == VK_SUCCESS);

    // Update descriptor sets (bind buffers)
    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    for (int i = 0; i < 2; i++) {
      VkDescriptorBufferInfo descriptor{
          .buffer = (i == 0 ? paramsBuffer : outputLossBuffer),
          .offset = 0,
          .range = (i == 0 ? sizeof(GLSLParameters) : sizeof(OutputLoss))};
      descriptorBufferInfos.push_back(descriptor);
    }
    std::vector<VkWriteDescriptorSet> descriptorWrites;
    for (int i = 0; i < 2; i++) {
      VkWriteDescriptorSet writeDescriptorSet{
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = vulkan->descriptorSet,
          .dstBinding = (i == 0 ? (uint)0 : (uint)6),
          .dstArrayElement = 0,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &descriptorBufferInfos.at(i)};
      descriptorWrites.push_back(writeDescriptorSet);
    }
    vkUpdateDescriptorSets(vulkan->logicalDevice,
                           static_cast<uint32_t>(descriptorWrites.size()),
                           descriptorWrites.data(), 0, nullptr);

    // Create shader module
    VkShaderModuleCreateInfo createInfoSM = {};
    createInfoSM.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfoSM.codeSize = computeShaderCode->size() * sizeof(uint32_t);
    createInfoSM.pCode = computeShaderCode->data();
    VkShaderModule computeShaderModule;
    result = vkCreateShaderModule(vulkan->logicalDevice, &createInfoSM, nullptr,
                                  &computeShaderModule);
    CHECK(result == VK_SUCCESS);

    // Create pipeline layout
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout setLayouts[] = {vulkan->descriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    result = vkCreatePipelineLayout(vulkan->logicalDevice, &pipelineLayoutInfo,
                                    nullptr, &pipelineLayout);
    CHECK(result == VK_SUCCESS);

    // Create compute pipeline
    VkPipeline computePipeline;
    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = computeShaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = 0;
    result = vkCreateComputePipelines(vulkan->logicalDevice, VK_NULL_HANDLE, 1,
                                      &pipelineInfo, nullptr, &computePipeline);
    CHECK(result == VK_SUCCESS);

    // Create command pool
    VkCommandPoolCreateInfo poolInfoCMD = {};
    poolInfoCMD.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfoCMD.queueFamilyIndex = vulkan->queueComputeIndex;
    poolInfoCMD.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    result = vkCreateCommandPool(vulkan->logicalDevice, &poolInfoCMD, nullptr,
                                 &vulkan->commandPool);
    CHECK(result == VK_SUCCESS);

    // Create command buffer pool
    VkCommandBufferAllocateInfo allocInfoCBP = {};
    allocInfoCBP.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfoCBP.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfoCBP.commandPool = vulkan->commandPool;
    allocInfoCBP.commandBufferCount = 1;
    vulkan->commandBufferPool = std::vector<VkCommandBuffer>(1);
    result = vkAllocateCommandBuffers(vulkan->logicalDevice, &allocInfoCBP,
                                      vulkan->commandBufferPool.data());
    CHECK(result == VK_SUCCESS);

    // Copy data to buffer memory
    GLSLParameters params{
        .learning_rate = 0.65f, .error_min = 0.1f, .error_max = 0.9f};
    void *paramsData;
    result = vkMapMemory(vulkan->logicalDevice, paramsBufferMemory, 0,
                         sizeof(GLSLParameters), 0, &paramsData);
    CHECK(result == VK_SUCCESS);
    memcpy(paramsData, &params, sizeof(GLSLParameters));
    vkUnmapMemory(vulkan->logicalDevice, paramsBufferMemory);

    // Run the shader
    VkCommandBuffer commandBuffer = vulkan->commandBufferPool.back();
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline);
    VkDescriptorSet descriptorSets[] = {vulkan->descriptorSet};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipelineLayout, 0, 1, descriptorSets, 0, nullptr);
    vkCmdDispatch(commandBuffer, 1, 1, 1);
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(vulkan->queueCompute, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkan->queueCompute);

    // Read data from buffer
    OutputLoss outputLoss{.loss = 0.0f};
    void *outputData;
    result = vkMapMemory(vulkan->logicalDevice, outputLossBufferMemory, 0,
                         sizeof(OutputLoss), 0, &outputData);
    CHECK(result == VK_SUCCESS);
    outputLoss.loss = *reinterpret_cast<float *>(outputData);
    vkUnmapMemory(vulkan->logicalDevice, outputLossBufferMemory);
    CHECK(outputLoss.loss == doctest::Approx(params.learning_rate * 2));

    // Cleanup
    vkDestroyBuffer(vulkan->logicalDevice, paramsBuffer, nullptr);
    vkFreeMemory(vulkan->logicalDevice, paramsBufferMemory, nullptr);
    vkDestroyBuffer(vulkan->logicalDevice, outputLossBuffer, nullptr);
    vkFreeMemory(vulkan->logicalDevice, outputLossBufferMemory, nullptr);
    vkDestroyShaderModule(vulkan->logicalDevice, computeShaderModule, nullptr);
    vkDestroyPipeline(vulkan->logicalDevice, computePipeline, nullptr);
    vkDestroyPipelineLayout(vulkan->logicalDevice, pipelineLayout, nullptr);
    builder.clear();
  }

  SUBCASE("Test various") {
    CHECK(sizeof(GLSLNeuron) ==
          (2 * sizeof(uint) + MAX_NEIGHBORS * sizeof(GLSLNeighbor) +
           sizeof(std::vector<std::vector<cv::Vec4f>>)));
  }
}