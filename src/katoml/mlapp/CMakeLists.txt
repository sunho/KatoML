set(SOURCE_LIST
  mnist_loader.hpp
  mlapp.hpp
  mlapp.cpp
)

add_library(mlapp STATIC ${SOURCE_LIST})
target_include_directories(mlapp PUBLIC ../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)
target_link_libraries(mlapp PUBLIC mlcompiler mltensor)