set(SOURCE_LIST
  errors.hpp
  mlsupport.cpp
  mlsupport.hpp
)

add_library(mlsupport STATIC ${SOURCE_LIST})
target_include_directories(mlsupport PUBLIC ../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)