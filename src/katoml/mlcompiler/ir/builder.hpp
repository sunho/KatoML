#pragma once
#include "katoml/mlcompiler/ir/types.hpp"
#include "katoml/mlsupport/errors.hpp"
#include "katoml/mltensor/core.hpp"
#include "value.hpp"
#include "node.hpp"

namespace katoml {
namespace compiler {
namespace ir {

class Builder {
public:
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) static inline Value OP(ARGS);
#include "ir.inc"
#undef DECL_NODE
static Value build_binary(ir::Opcode opcode,ir::Value lhs,ir::Value rhs) {
  DataType ldatatype = lhs.get_datatype(), rdatatype = rhs.get_datatype();
  TYPE_CHECK_OR_THROW(
    tensor::can_broadcast_shape(lhs.get_shape(), rhs.get_shape()) && 
    lhs.get_element_type() == rhs.get_element_type(), 
    opcode_to_string(opcode), ldatatype, rdatatype);
  tensor::Shape res_shape = calculate_broadcast_shape(ldatatype.get_shape(), rdatatype.get_shape());
  return NodePtr(new Node(opcode, DataType(ldatatype.get_element_type(), res_shape), {lhs, rhs}));
}
static Value build_unary_same_type(ir::Opcode opcode,ir::Value val) {
  return NodePtr(new Node(opcode, val.get_datatype(), {val}));
}
static Value build_reduce_same_type(ir::Opcode opcode,ir::Value val, ir::IntListValue axis) {
  axis = val.get_datatype().get_shape().normalize_axis(axis.as_int_list());
  DataType datatype = val.get_datatype();
  tensor::Shape res_shape = datatype.get_shape().reduce(axis.as_int_list());
  return NodePtr(new Node(opcode, DataType(datatype.get_element_type(), res_shape), {val, axis}));
}
};

Value Builder::Add(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Add, lhs, rhs);
}

Value Builder::Sub(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Sub, lhs, rhs);
}

Value Builder::Mul(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Mul, lhs, rhs);
}

Value Builder::Div(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Div, lhs, rhs);
}

Value Builder::Max(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Max, lhs, rhs);
}

Value Builder::Min(ir::Value lhs,ir::Value rhs) {
  return build_binary(Opcode::Min, lhs, rhs);
}

Value Builder::Log(ir::Value val) {
  return build_unary_same_type(Opcode::Log, val);
}

Value Builder::Exp(ir::Value val) {
  return build_unary_same_type(Opcode::Exp, val);
}

Value Builder::Neg(ir::Value val) {
  return build_unary_same_type(Opcode::Neg, val);
}

Value Builder::MatMul(ir::Value lhs,ir::Value rhs) {
  DataType ldatatype = lhs.get_datatype(), rdatatype = rhs.get_datatype();
  TYPE_CHECK_OR_THROW(tensor::can_matmul(ldatatype, rdatatype), "matmul", ldatatype, rdatatype);
  tensor::Shape res_shape = tensor::calculate_matmul_shape(ldatatype.get_shape(), rdatatype.get_shape());
  return NodePtr(new Node(Opcode::MatMul, DataType(ldatatype.get_element_type(), res_shape), {lhs, rhs}));
}

Value Builder::ReduceSum(ir::Value val, ir::IntListValue axis) {
  return build_reduce_same_type(Opcode::ReduceSum, val, axis);
}

Value Builder::ReduceMean(ir::Value val, ir::IntListValue axis) {
  return build_reduce_same_type(Opcode::ReduceMean, val, axis);
}

Value Builder::SoftMax(ir::Value val) {
  return build_unary_same_type(Opcode::SoftMax, val);
}

Value Builder::LogSoftMax(ir::Value val) {
  return build_unary_same_type(Opcode::LogSoftMax, val);
}

}
}
}