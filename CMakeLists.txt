cmake_minimum_required(VERSION 3.18)

option(ENABLE_ASAN "turn on address santiziers and various debug flags" OFF)
option(ENABLE_TESTS "enable unit tests" OFF)
option(ENABLE_EXAMPLES "enable examples" ON)

if (ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -g)
  add_link_options(-fsanitize=address,undefined -g)
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(katoml)

add_subdirectory(external)

if (ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
if (ENABLE_TESTS)
  add_subdirectory(tests)
endif()

add_subdirectory(src/katoml)