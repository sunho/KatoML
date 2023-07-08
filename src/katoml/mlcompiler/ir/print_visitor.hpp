#pragma once

#include "ir.hpp"
#include "value.hpp"
#include "../tensor.hpp"
#include "../utils/format.hpp"
#include "visitor.hpp"

namespace katoml {
namespace compiler {
namespace ir {

struct PrinterVisitor {
  std::ostream& os;
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) inline void OP(ARGS);
#include "ir.inc"
#undef DECL_NODE
};

inline std::ostream& operator<<(std::ostream& os, const Node& node) {
  PrinterVisitor visitor{os};
  NodeVisitorCaller<void, PrinterVisitor>().call(visitor, node);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Var& var) {
  os << "Var(";
  if (var.has_tensor()) {
    os << var.get_tensor();
  } else {
    os << "null";
  }
  os << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Value& value) {
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

void PrinterVisitor::Add(ir::Value lhs, ir::Value rhs) {
  os << format("Add({}, {})", lhs, rhs);
}

void PrinterVisitor::Sub(ir::Value lhs, ir::Value rhs) {
  os << format("Sub({}, {})", lhs, rhs);
}

void PrinterVisitor::Mul(ir::Value lhs, ir::Value rhs) {
  os << format("Mul({}, {})", lhs, rhs);
}

void PrinterVisitor::Div(ir::Value lhs, ir::Value rhs) {
  os << format("Div({}, {})", lhs, rhs);
}

void PrinterVisitor::Max(ir::Value lhs, ir::Value rhs) {
  os << format("Max({}, {})", lhs, rhs);
}

void PrinterVisitor::Min(ir::Value lhs, ir::Value rhs) {
  os << format("Min({}, {})", lhs, rhs);
}

void PrinterVisitor::SoftMax(ir::Value val) {
  os << format("SoftMax({})", val);
}

void PrinterVisitor::LogSoftMax(ir::Value val) {
  os << format("LogSoftMax({})", val);
}

void PrinterVisitor::Neg(ir::Value val) {
  os << format("Neg({})", val);
}

void PrinterVisitor::Log(ir::Value val) {
  os << format("Log({})", val);
}

void PrinterVisitor::ReduceSum(ir::Value val, ir::IntListValue axis) {
  os << format("ReduceSum({}, axis={})", val, axis);
}

void PrinterVisitor::ReduceMean(ir::Value val, ir::IntListValue axis) {
  os << format("ReduceMean({}, axis={})", val, axis);
}

void PrinterVisitor::MatMul(ir::Value lhs, ir::Value rhs) {
  os << format("MatMul({}, {})", lhs, rhs);
}

}
}
}