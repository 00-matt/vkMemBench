cmake_minimum_required(VERSION 3.28)
project(vkMemBench CXX)

find_package(Vulkan 1.3 REQUIRED COMPONENTS glslangValidator)

add_executable(vkmembench src/vkcontext.cc src/vkmembench.cc)
set_target_properties(vkmembench PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
target_link_libraries(vkmembench PRIVATE Vulkan::Vulkan)

add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/test.spv"
  COMMAND Vulkan::glslangValidator --target-env vulkan1.3 "${CMAKE_CURRENT_SOURCE_DIR}/src/test.comp" -o "${CMAKE_BINARY_DIR}/test.spv"
  DEPENDS src/test.comp
  VERBATIM
)
set_property(TARGET vkmembench APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/test.spv")
