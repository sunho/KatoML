#pragma once
#include "../tensor.hpp"

namespace katoml {
namespace compiler {
namespace ir {

#define DECL_NODE(name, ...) name,
enum class Opcode {
#include "ir.inc"
};
#undef DECL_NODE

static inline std::string opcode_to_string(Opcode opcode) {
  switch(opcode) {
#define DECL_NODE(name, ...) case Opcode::name: return #name;
#include "ir.inc"
#undef DECL_NODE
  }
}

class Node;

using DataType = tensor::DataType;
using NodePtr = std::shared_ptr<Node>;
using IntList = std::vector<int>;

}
}
}