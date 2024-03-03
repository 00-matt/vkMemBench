// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include <vulkan/vulkan_core.h>

class Context;

class Buffer {
  friend Context;

  const Context &m_context;
  const VkBuffer m_handle;
  const std::uint32_t m_size;
  std::optional<VkDeviceMemory> m_allocation;
  bool m_mapped;

  Buffer(const Context &context, VkBuffer handle, std::uint32_t size)
      : m_context(context), m_handle(handle), m_size(size), m_mapped(false) {}

public:
  Buffer(const Buffer &) = delete;
  Buffer(Buffer &&) = delete;
  ~Buffer();

  VkDeviceMemory allocate(std::uint32_t memory_type_mask);
  std::span<std::uint8_t> mmap();
  void munmap();

  VkBuffer handle() const { return m_handle; }
  std::uint32_t size() const { return m_size; }
  std::optional<VkDeviceMemory> allocation() const { return m_allocation; }
};

class Fence {
  friend Context;

  const Context &m_context;
  const VkFence m_fence;

public:
  Fence(const Context &context, VkFence fence) : m_context(context), m_fence(fence) {}
  Fence(const Fence &) = delete;
  Fence(Fence &&) = delete;
  ~Fence();

  void wait() const;
  void reset() const;

  VkFence handle() const { return m_fence; }
};

class Context {
  friend Buffer;

  VkInstance m_instance = nullptr;
  VkPhysicalDevice m_physical_device = nullptr;
  VkQueue m_compute_queue = nullptr;
  VkDevice m_device = nullptr;
  VkCommandPool m_compute_command_pool = nullptr;

  void create_instance(bool validation_enabled);
  void create_device();

  std::optional<std::uint32_t> find_memory_type(std::uint32_t flags) const;

public:
  Context(bool validation_enabled);
  Context(const Context &) = delete;
  Context(Context &&) = delete;
  ~Context();

  Buffer create_buffer(std::uint32_t size, std::uint32_t usage) const;
  Fence create_fence() const;

  VkInstance instance() const { return m_instance; }
  VkPhysicalDevice physical_device() const { return m_physical_device; }
  VkDevice device() const { return m_device; }
  VkCommandPool compute_command_pool() const { return m_compute_command_pool; }
  VkQueue compute_queue() const { return m_compute_queue; }
};
