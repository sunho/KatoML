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

template<class Backend>
class Value {
public:
  Value() = default;
  Value(const TTensorPtr& tensor) : 
    type(ValueType::Tensor), impl(tensor) { }
  Value(const IntList& list) :
    type(ValueType::IntList), impl(list) {}
  Value(const std::string& str) :
    type(ValueType::Str), impl(str) {}
  Value(const NodePtr<Backend>& node) :
    type(ValueType::Node), impl(node) {}
  Value(const VarPtr<Backend>& var) :
    type(ValueType::Var), impl(var) {}
  ValueType get_type() const { return type; }
  inline DataType get_datatype() const;
  bool is_tensor() const { return type == ValueType::Tensor || is_node() || is_var(); }
  bool is_var() const { return type == ValueType::Var; }
  bool is_node() const { return type == ValueType::Node; }
  std::string as_str() const { 
    assert(type == ValueType::Str);
    return std::get<std::string>(impl);
  }
  IntList as_int_list() const {
    assert(type == ValueType::IntList);
    return std::get<IntList>(impl);
  }
  TTensorPtr as_tensor() const {
    assert(type == ValueType::Tensor);
    return std::get<TTensorPtr>(impl);
  }
  NodePtr<Backend> as_node() const {
    assert(type == ValueType::Node);
    return std::get<NodePtr<Backend>>(impl);
  }
  VarPtr<Backend> as_var() const {
    assert(type == ValueType::Var);
    return std::get<VarPtr<Backend>>(impl);
  }
private:
  using ValueImpl = std::variant<TTensorPtr, IntList, std::string, NodePtr<Backend>, VarPtr<Backend>>;
  ValueType type { ValueType::Void };
  ValueImpl impl { };
};

template<class Backend, class T, ValueType VT>
class TypedValue : public Value<Backend> {
public:
  TypedValue() = delete;
  TypedValue(const T& value) : Value<Backend>(value) {}
  TypedValue(const Value<Backend>& value) : Value<Backend>(value) {
    assert(value.get_type() == VT);
  }
};

template<class Backend>
using StrValue = TypedValue<Backend, std::string, ValueType::Str>;
template<class Backend>
using IntListValue = TypedValue<Backend, IntList, ValueType::IntList>;
template<class Backend>
using TensorValue = TypedValue<Backend, TTensorPtr, ValueType::Tensor>;
template<class Backend>
using NodeValue = TypedValue<Backend, NodePtr<Backend>, ValueType::Node>;
template<class Backend>
using VarValue = TypedValue<Backend, VarPtr<Backend>, ValueType::Var>;

}
}
}