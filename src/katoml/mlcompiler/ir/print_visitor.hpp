#pragma once
#include <fmt/format.h>
#include <fmt/ostream.h>

#include "ir.hpp"
#include "value.hpp"
#include "../tensor.hpp"
#include "visitor.hpp"

namespace katoml {
namespace compiler {
namespace ir {

template<class Backend>
struct PrinterVisitor {
  std::ostream& os;
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) inline void OP(ARGS);
#include "ir.inc"
#undef DECL_NODE
};

template<class Backend>
inline std::ostream& operator<<(std::ostream& os, const Node<Backend>& node) {
  PrinterVisitor<Backend> visitor{os};
  NodeVisitorCaller<Backend, void, PrinterVisitor<Backend>>().call(visitor, node);
  return os;
}

template<class Backend>
inline std::ostream& operator<<(std::ostream& os, const Var<Backend>& var) {
  os << "Var(";
  os << var.get_tensor();
  os << ")";
  return os;
}

template<class Backend>
inline std::ostream& operator<<(std::ostream& os, const Value<Backend>& value) {
  switch (value.get_type()) {
  case ValueType::Void:
    os << "<Void>";
    break;
  case ValueType::IntList: {
    os << "[";
    auto list = value.as_int_list();
    for (size_t i = 0; i < list.size(); i++) {
      if (i == list.size() - 1) {
        os << list[i] << "";
      } else {
        os << list[i] << ", ";
      }
    }
    os << "]";
    break;
  }
  case ValueType::Str: {
    os << "<" << value.as_str() << ">";
    break;
  }
  case ValueType::Tensor: {
    os << *value.as_tensor();
    break;
  }
  case ValueType::Node: {
    os << *value.as_node();
    break;
  }
  case ValueType::Var: {
    os << *value.as_var();
    break;
  }
  }
  return os;
}


template<class Backend>
void PrinterVisitor<Backend>::Add(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Add({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::Sub(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Sub({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::Mul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Mul({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::Div(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Div({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::Max(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Max({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::Min(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("Min({}, {})", lhs, rhs);
}

template<class Backend>
void PrinterVisitor<Backend>::SoftMax(ir::Value<Backend> val) {
  os << fmt::format("SoftMax({})", val);
}

template<class Backend>
void PrinterVisitor<Backend>::LogSoftMax(ir::Value<Backend> val) {
  os << fmt::format("LogSoftMax({})", val);
}

template<class Backend>
void PrinterVisitor<Backend>::Neg(ir::Value<Backend> val) {
  os << fmt::format("Neg({})", val);
}

template<class Backend>
void PrinterVisitor<Backend>::Log(ir::Value<Backend> val) {
  os << fmt::format("Log({})", val);
}

template<class Backend>
void PrinterVisitor<Backend>::ReduceSum(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  os << fmt::format("ReduceSum({}, axis={})", val, axis);
}

template<class Backend>
void PrinterVisitor<Backend>::ReduceMean(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  os << fmt::format("ReduceMean({}, axis={})", val, axis);
}

template<class Backend>
void PrinterVisitor<Backend>::MatMul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  os << fmt::format("MatMul({}, {})", lhs, rhs);
}

}
}
}