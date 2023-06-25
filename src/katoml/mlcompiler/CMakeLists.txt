set(SOURCE_LIST
  device.hpp
  mlcompiler.hpp
  mlcompiler.cpp
  tensor.hpp
  ir/builder.hpp
  ir/ir.hpp
  ir/ir.inc
  ir/node.hpp
  ir/print_visitor.hpp
  ir/types.hpp
  ir/value.hpp
  ir/visitor.hpp
  graph/graph.hpp
)

add_library(mlcompiler STATIC ${SOURCE_LIST})
target_include_directories(mlcompiler PUBLIC ../.. ${CMAKE_CURRENT_BINARY_DIR}/../..)
target_link_libraries(mlcompiler PUBLIC mltensor)