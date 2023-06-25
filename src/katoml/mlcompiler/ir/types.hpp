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

template<class Backend>
class Node;

using DataType = tensor::DataType;
template<class Backend>
using NodePtr = std::shared_ptr<Node<Backend>>;
using IntList = std::vector<int>;

}
}
}