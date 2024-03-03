cmake_minimum_required(VERSION 3.28)
project(vkMemBench CXX)

find_package(Vulkan REQUIRED)

add_executable(vkmembench src/vkcontext.cc src/vkmembench.cc)
set_target_properties(vkmembench PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
target_link_libraries(vkmembench PRIVATE Vulkan::Vulkan)
