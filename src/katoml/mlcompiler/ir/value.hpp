#pragma once
#include "katoml/mlcompiler/tensor.hpp"
#include "types.hpp"
#include "var.hpp"

namespace katoml {
namespace compiler {
namespace ir {

enum class ValueType {
  Void,
  Tensor,
  IntList,
  Str,
  Node,
  Var,
};

class Value {
public:
  Value() = default;
  Value(const TensorPtr& tensor) : 
    type(ValueType::Tensor), impl(tensor) { }
  Value(const IntList& list) :
    type(ValueType::IntList), impl(list) {}
  Value(const std::string& str) :
    type(ValueType::Str), impl(str) {}
  Value(const NodePtr& node) :
    type(ValueType::Node), impl(node) {}
  Value(const VarPtr& var) :
    type(ValueType::Var), impl(var) {}
  ValueType get_type() const { return type; }
  inline DataType get_datatype() const;
  bool is_tensor() const { return type == ValueType::Tensor || is_node() || is_var(); }
  bool is_var() const { return type == ValueType::Var; }
  bool is_node() const { return type == ValueType::Node; }
  std::string as_str() const { 
    ASSERT(type == ValueType::Str, "tried to unwrap with wrong value type");
    return std::get<std::string>(impl);
  }
  IntList as_int_list() const {
    ASSERT(type == ValueType::IntList, "tried to unwrap with wrong value type");
    return std::get<IntList>(impl);
  }
  TensorPtr as_tensor() const {
    ASSERT(type == ValueType::Tensor, "tried to unwrap with wrong value type");
    return std::get<TensorPtr>(impl);
  }
  NodePtr as_node() const {
    ASSERT(type == ValueType::Node, "tried to unwrap with wrong value type");
    return std::get<NodePtr>(impl);
  }
  VarPtr as_var() const {
    ASSERT(type == ValueType::Var, "tried to unwrap with wrong value type");
    return std::get<VarPtr>(impl);
  }
private:
  using ValueImpl = std::variant<TensorPtr, IntList, std::string, NodePtr, VarPtr>;
  ValueType type { ValueType::Void };
  ValueImpl impl { };
};

template<class T, ValueType VT>
class TypedValue : public Value {
public:
  TypedValue() = delete;
  TypedValue(const T& value) : Value(value) {}
  TypedValue(const Value& value) : Value(value) {
    assert(value.get_type() == VT);
  }
};

using StrValue = TypedValue<std::string, ValueType::Str>;
using IntListValue = TypedValue<IntList, ValueType::IntList>;
using TensorValue = TypedValue<TensorPtr, ValueType::Tensor>;
using NodeValue = TypedValue<NodePtr, ValueType::Node>;
using VarValue = TypedValue<VarPtr, ValueType::Var>;

}
}
}