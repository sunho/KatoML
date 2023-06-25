#pragma once
#include "katoml/mlcompiler/ir/types.hpp"
#include "value.hpp"
#include "node.hpp"

namespace katoml {
namespace compiler {
namespace ir {

template<class Backend>
class Builder {
public:
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) static inline Value<Backend> OP(ARGS);
#include "ir.inc"
#undef DECL_NODE
static Value<Backend> build_binary(ir::Opcode opcode, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  DataType ldatatype = lhs.get_datatype(), rdatatype = rhs.get_datatype();
  assert(ldatatype.get_element_type() == rdatatype.get_element_type());
  assert(tensor::can_broadcast_shape(ldatatype.get_shape(), rdatatype.get_shape()));
  tensor::Shape res_shape = calculate_broadcast_shape(ldatatype.get_shape(), rdatatype.get_shape());
  return NodePtr<Backend>(new Node<Backend>(opcode, DataType(ldatatype.get_element_type(), res_shape), {lhs, rhs}));
}
static Value<Backend> build_unary_same_type(ir::Opcode opcode, ir::Value<Backend> val) {
  return NodePtr<Backend>(new Node<Backend>(opcode, val.get_datatype(), {val}));
}
static Value<Backend> build_reduce_same_type(ir::Opcode opcode, ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  DataType datatype = val.get_datatype();
  tensor::Shape res_shape = datatype.get_shape().reduce(axis.as_int_list());
  return NodePtr<Backend>(new Node<Backend>(opcode, DataType(datatype.get_element_type(), res_shape), {val, axis}));
}
};

template<class Backend>
DataType Value<Backend>::get_datatype() const { 
  assert(is_tensor());
  if (is_node()) { return as_node()->get_datatype(); }
  if (is_var()) { return as_var()->get_datatype(); }
  return as_tensor()->get_datatype();
}

template<class Backend>
Value<Backend> Builder<Backend>::Add(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Add, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Sub(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Sub, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Mul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Mul, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Div(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Div, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Max(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Max, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Min(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return build_binary(Opcode::Min, lhs, rhs);
}

template<class Backend>
Value<Backend> Builder<Backend>::Log(ir::Value<Backend> val) {
  return build_unary_same_type(Opcode::Log, val);
}

template<class Backend>
Value<Backend> Builder<Backend>::Neg(ir::Value<Backend> val) {
  return build_unary_same_type(Opcode::Neg, val);
}

template<class Backend>
Value<Backend> Builder<Backend>::MatMul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  DataType ldatatype = lhs.get_datatype(), rdatatype = rhs.get_datatype();
  assert(ldatatype.get_element_type() == rdatatype.get_element_type());
  assert(ldatatype.get_shape()[1] == rdatatype.get_shape()[0]);
  tensor::Shape res_shape = tensor::calculate_matmul_shape(ldatatype.get_shape(), rdatatype.get_shape());
  return NodePtr<Backend>(new Node<Backend>(Opcode::MatMul, DataType(ldatatype.get_element_type(), res_shape), {lhs, rhs}));
}

template<class Backend>
Value<Backend> Builder<Backend>::ReduceSum(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  return build_reduce_same_type(Opcode::ReduceSum, val, axis);
}

template<class Backend>
Value<Backend> Builder<Backend>::ReduceMean(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  return build_reduce_same_type(Opcode::ReduceMean, val, axis);
}

template<class Backend>
Value<Backend> Builder<Backend>::SoftMax(ir::Value<Backend> val) {
  return build_unary_same_type(Opcode::SoftMax, val);
}

template<class Backend>
Value<Backend> Builder<Backend>::LogSoftMax(ir::Value<Backend> val) {
  return build_unary_same_type(Opcode::LogSoftMax, val);
}

}
}
}