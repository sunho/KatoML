cmake_minimum_required(VERSION 3.18)

if (ASAN)
  add_compile_options(-fsanitize=address,undefined -g)
  add_link_options(-fsanitize=address,undefined -g)
endif()

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ version selection")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(katoml)

add_subdirectory(external)
add_subdirectory(examples)
add_subdirectory(tests)
add_subdirectory(src/katoml)