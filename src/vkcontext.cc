// SPDX-License-Identifier: GPL-3.0-or-later
#include "vkcontext.hh"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan_core.h>

Buffer::~Buffer() {
  if (m_allocation) {
    vkFreeMemory(m_context.device(), m_allocation.value(), nullptr);
  }
  vkDestroyBuffer(m_context.device(), m_handle, nullptr);
}

VkDeviceMemory Buffer::allocate(std::uint32_t memory_type_mask) {
  std::optional<std::uint32_t> memory_type = m_context.find_memory_type(memory_type_mask);
  if (!memory_type) {
    throw std::runtime_error("unable to find memory type");
  }

  VkMemoryAllocateFlagsInfo alloc_flags_info{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
  };
  VkMemoryAllocateInfo alloc_ci{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &alloc_flags_info,
      .allocationSize = m_size,
      .memoryTypeIndex = memory_type.value(),
  };
  VkDeviceMemory buffer_memory;
  if (vkAllocateMemory(m_context.device(), &alloc_ci, nullptr, &buffer_memory) != VK_SUCCESS) {
    throw std::runtime_error("unable to allocate buffer");
  }
  m_allocation.emplace(buffer_memory);

  if (vkBindBufferMemory(m_context.device(), m_handle, buffer_memory, 0) != VK_SUCCESS) {
    throw std::runtime_error("unable to bind buffer");
  }

  return buffer_memory;
}

std::span<std::uint8_t> Buffer::mmap() {
  assert(m_allocation);
  assert(!m_mapped);

  void *ptr;
  if (vkMapMemory(m_context.device(), m_allocation.value(), 0, m_size, 0, &ptr) != VK_SUCCESS) {
    throw std::runtime_error("unable to map memory");
  }
  m_mapped = true;
  return {reinterpret_cast<std::uint8_t *>(ptr), m_size};
}

void Buffer::munmap() {
  assert(m_mapped);
  m_mapped = false;
  vkUnmapMemory(m_context.device(), m_allocation.value());
}

Fence::~Fence() { vkDestroyFence(m_context.device(), m_fence, nullptr); }

void Fence::wait() const {
  // TODO: timeout parameter
  if (vkWaitForFences(m_context.device(), 1, &m_fence, true, std::numeric_limits<std::uint64_t>::max()) != VK_SUCCESS) {
    throw std::runtime_error("unable to wait for fence");
  }
}

void Fence::reset() const {
  if (vkResetFences(m_context.device(), 1, &m_fence) != VK_SUCCESS) {
    throw std::runtime_error("unable to reset fence");
  }
}

Context::Context(bool validation_enabled) {
  create_instance(validation_enabled);
  create_device();
}

Context::~Context() {
  if (m_compute_command_pool) {
    vkDestroyCommandPool(m_device, m_compute_command_pool, nullptr);
  }
  if (m_device) {
    vkDestroyDevice(m_device, nullptr);
  }
  if (m_instance) {
    vkDestroyInstance(m_instance, nullptr);
  }
}

void Context::create_instance(bool validation_enabled) {
  // TODO: check if validation layers present
  VkApplicationInfo application_info{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_MAKE_VERSION(1, 3, 0),
  };
  const char *validation_layer_name = "VK_LAYER_KHRONOS_validation";
  VkInstanceCreateInfo instance_ci{
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pNext = nullptr,
      .pApplicationInfo = &application_info,
      .enabledLayerCount = static_cast<std::uint32_t>(validation_enabled ? 1 : 0),
      .ppEnabledLayerNames = validation_enabled ? &validation_layer_name : nullptr,
  };
  if (vkCreateInstance(&instance_ci, nullptr, &m_instance)) {
    throw std::runtime_error("unable to create vulkan instance");
  }
}

void Context::create_device() {
  // Select physical device
  std::uint32_t physical_device_count = 1;
  VkResult result = vkEnumeratePhysicalDevices(m_instance, &physical_device_count, &m_physical_device);
  if ((result != VK_SUCCESS && result != VK_INCOMPLETE) || physical_device_count != 1) {
    throw std::runtime_error("unable to find physical device");
  }

  // Find compute queue
  std::uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physical_device, &queue_family_count, queue_families.data());
  bool found_queue = false;
  const float queue_priority = 1.f;
  VkDeviceQueueCreateInfo compute_queue_ci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pQueuePriorities = &queue_priority,
  };
  for (std::size_t i = 0; i < queue_families.size(); i++) {
    if (queue_families[i].queueFlags != (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) {
      continue;
    }
    found_queue = true;
    compute_queue_ci.queueFamilyIndex = i;
    compute_queue_ci.queueCount = 1;
    break;
  }
  if (!found_queue) {
    throw std::runtime_error("unable to find compute-capable queue");
  }

  // Create logical device
  VkPhysicalDeviceVulkan12Features device_12_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .hostQueryReset = true,
      .bufferDeviceAddress = true,
  };
  VkPhysicalDeviceVulkan13Features device_13_features{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &device_12_features,
      .synchronization2 = true,
  };
  VkDeviceCreateInfo device_ci{
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &device_13_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &compute_queue_ci,
  };
  if (vkCreateDevice(m_physical_device, &device_ci, nullptr, &m_device) != VK_SUCCESS) {
    throw std::runtime_error("unable to create device");
  }

  // TODO: check for error
  vkGetDeviceQueue(m_device, compute_queue_ci.queueFamilyIndex, 0, &m_compute_queue);

  // Create command pool for compute queue
  VkCommandPoolCreateInfo command_pool_ci{
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = compute_queue_ci.queueFamilyIndex,
  };
  vkCreateCommandPool(m_device, &command_pool_ci, nullptr, &m_compute_command_pool);
}

std::optional<std::uint32_t> Context::find_memory_type(std::uint32_t flags) const {
  VkPhysicalDeviceMemoryProperties properties;
  vkGetPhysicalDeviceMemoryProperties(m_physical_device, &properties);
  for (std::uint32_t i = 0; i < properties.memoryTypeCount; i++) {
    // if ((properties.memoryTypes[i].propertyFlags & flags) == flags) {
    if (properties.memoryTypes[i].propertyFlags == flags) {
      return i;
    }
  }
  return {};
}

Buffer Context::create_buffer(std::uint32_t size, std::uint32_t usage) const {
  VkBufferCreateInfo buffer_ci{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
  };
  VkBuffer buffer;
  if (vkCreateBuffer(m_device, &buffer_ci, nullptr, &buffer) != VK_SUCCESS) {
    throw std::runtime_error("unable to allocate buffer");
  }
  return {*this, buffer, size};
}

Fence Context::create_fence() const {
  VkFenceCreateInfo fence_ci{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
  };
  VkFence fence;
  // TODO: check for error
  vkCreateFence(m_device, &fence_ci, nullptr, &fence);
  return {*this, fence};
}
