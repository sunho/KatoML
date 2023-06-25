set(SOURCE_LIST
  core.hpp
  cpu_backend.hpp
  iter_utils.hpp
  mltensor.cpp
  mltensor.hpp
  types.hpp
)

add_library(mltensor STATIC ${SOURCE_LIST})
target_include_directories(mltensor PUBLIC ../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)