// SPDX-License-Identifier: GPL-3.0-or-later
#include "vkcontext.hh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <span>

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan_core.h>

void copy_benchmark(Context &context, std::uint64_t buffer_size) {
  VkPhysicalDeviceProperties physical_properties;
  vkGetPhysicalDeviceProperties(context.physical_device(), &physical_properties);
  // nanos in one timestamp tick
  float period = physical_properties.limits.timestampPeriod;

  Buffer src = context.create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  src.allocate(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  std::span<std::uint8_t> data = src.mmap();
  std::fill(data.begin(), data.end(), 0xff);

  Buffer dst = context.create_buffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  dst.allocate(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkQueryPoolCreateInfo query_pool_ci{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = 2,
  };
  VkQueryPool query_pool;
  vkCreateQueryPool(context.device(), &query_pool_ci, nullptr, &query_pool);

  Fence transfer_fence = context.create_fence();

  // Allocate command buffer
  VkCommandBufferAllocateInfo command_buffer_ai{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = context.compute_command_pool(),
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(context.device(), &command_buffer_ai, &command_buffer);

  // Record command buffer with single copy command
  VkCommandBufferBeginInfo command_buffer_begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);
  // Could split the copy up into multiple smaller copies, but this
  // didn't seem to make a performance difference with RADV (which
  // doesn't use hardware transfer queues).
  VkBufferCopy copy{
      .srcOffset = 0,
      .dstOffset = 0,
      .size = buffer_size,
  };
  vkCmdResetQueryPool(command_buffer, query_pool, 0, 2);
  vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_NONE, query_pool, 0);
  vkCmdCopyBuffer(command_buffer, src.handle(), dst.handle(), 1, &copy);
  vkCmdWriteTimestamp2(command_buffer, VK_PIPELINE_STAGE_2_COPY_BIT, query_pool, 1);
  vkEndCommandBuffer(command_buffer);

  // Submit command buffer to queue
  VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
  };

  float total_seconds = 0;
  std::uint64_t total_bytes = 0;

  // This has the overhead of round trip latency between CPU and
  // GPU. Instead, could have multiple copy buffer commands and submit
  // those at the same time.
  for (int count = 0; count < 32; count++) {
    vkQueueSubmit(context.compute_queue(), 1, &submit_info, transfer_fence.handle());

    transfer_fence.wait();
    transfer_fence.reset();

    std::array<std::uint64_t, 2> timestamps;
    vkGetQueryPoolResults(context.device(), query_pool, 0, 2, timestamps.size() * sizeof(std::uint64_t),
                          timestamps.data(), sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    double seconds = static_cast<double>(timestamps[1] - timestamps[0]) * period / 1e9f;

    total_seconds += seconds;
    total_bytes += buffer_size;
  }

  std::cout << buffer_size / 1024 / 1024 << " MiB @ " << total_bytes / total_seconds / 1024 / 1024 << " MiB/sec\n";

  vkDestroyQueryPool(context.device(), query_pool, nullptr);
}

VkShaderModule load_shader(Context &context, const char *path) {
  // TODO: fix path
  std::ifstream shader_file(path, std::ios::binary | std::ios::ate);
  if (!shader_file) {
    throw std::runtime_error("unable to open compute shader binary");
  }
  std::size_t shader_size = shader_file.tellg();
  shader_file.seekg(0);
  std::vector<char> shader_data(shader_size);
  shader_file.read(shader_data.data(), shader_size);

  VkShaderModuleCreateInfo shader_ci{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = shader_size,
      .pCode = reinterpret_cast<std::uint32_t *>(shader_data.data()),
  };
  VkShaderModule shader;
  if (vkCreateShaderModule(context.device(), &shader_ci, nullptr, &shader) != VK_SUCCESS) {
    throw std::runtime_error("unable to create shader module");
  }
  return shader;
}

int main() {
  Context context(true);

  // std::cout << "host-to-device copy (compute queue)\n--------------------\n";
  // // TODO: Check memory size instead of assuming 1 GiB is available.
  // for (uint64_t i = 0; i < 11; i++) {
  //   copy_benchmark(context, 1024ull * 1024 * (1 << i));
  // }

  Buffer src =
      context.create_buffer(4 * 32, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  src.allocate(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  std::span<std::uint8_t> src_data = src.mmap();
  std::fill(src_data.begin(), src_data.end(), 0x80);
  std::cout << " in = 0x" << std::hex << static_cast<int>(src_data[0]) << '\n';
  VkBufferDeviceAddressInfo src_info{
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = src.handle(),
  };
  VkDeviceAddress src_address = vkGetBufferDeviceAddress(context.device(), &src_info);

  VkPushConstantRange push_constant_range{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = sizeof(VkDeviceAddress),
  };
  VkPipelineLayoutCreateInfo pipeline_layout_ci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &push_constant_range,
  };
  VkPipelineLayout pipeline_layout;
  vkCreatePipelineLayout(context.device(), &pipeline_layout_ci, nullptr, &pipeline_layout);

  VkShaderModule shader = load_shader(context, "test.spv");

  VkPipelineShaderStageCreateInfo pipeline_shader_stage_ci{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main",
  };
  VkComputePipelineCreateInfo compute_pipeline_ci{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = pipeline_shader_stage_ci,
      .layout = pipeline_layout,
  };
  VkPipeline compute_pipeline;
  vkCreateComputePipelines(context.device(), nullptr, 1, &compute_pipeline_ci, nullptr, &compute_pipeline);
  vkDestroyShaderModule(context.device(), shader, nullptr);

  // Allocate command buffer
  VkCommandBufferAllocateInfo command_buffer_ai{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = context.compute_command_pool(),
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
  };
  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(context.device(), &command_buffer_ai, &command_buffer);

  // Record command buffer with
  VkCommandBufferBeginInfo command_buffer_begin_info{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  };
  vkBeginCommandBuffer(command_buffer, &command_buffer_begin_info);

  vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline);
  vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress),
                     &src_address);
  vkCmdDispatch(command_buffer, 1, 1, 1);

  vkEndCommandBuffer(command_buffer);

  // Submit command buffer to queue
  VkSubmitInfo submit_info{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
  };

  Fence fence = context.create_fence();
  vkQueueSubmit(context.compute_queue(), 1, &submit_info, fence.handle());
  fence.wait();
  fence.reset();

  vkDestroyPipelineLayout(context.device(), pipeline_layout, nullptr);
  vkDestroyPipeline(context.device(), compute_pipeline, nullptr);

  std::cout << "out = 0x" << std::hex << static_cast<int>(src_data[0]) << '\n';
}
