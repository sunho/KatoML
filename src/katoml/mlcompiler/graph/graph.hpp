#pragma once

#include "../ir/ir.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/types.hpp"
#include <functional>
#include <initializer_list>

namespace katoml {
namespace compiler {

template <typename T> concept signed_type = std::is_signed_v<T>; 

class GraphDevice;

static inline std::vector<int> unwrap_axis(const std::vector<int>& axis, ir::Value value) {
  if (axis == tensor::AllAxis) {
    std::vector<int> axis_(value.get_datatype().get_shape().get_ndims());
    std::iota(std::begin(axis_), std::end(axis_), 0);
    return axis_;
  }
  return axis;
}

class Node final {
public:
  Node() = default;
  Node(GraphDevice& device, ir::Value value) : 
    device(&device), value(value) {}
  Node(GraphDevice& device, Tensor&& tensor) :
    device(&device), value(std::make_shared<Tensor>(std::move(tensor))) {}
  inline Node(GraphDevice& device, tensor::ElementType element_type, tensor::Constant val);
  Node(const Node& other) = default;
  Node& operator=(const Node& rhs) = default;

  #define BINOP_CPP(def_name, op_name) \
  friend Node def_name(Node lhs, Node rhs) { \
    return Node(*lhs.device, ir::Builder::op_name(lhs.value, rhs.value)); \
  }\
  friend Node def_name(tensor::Constant lhs, Node rhs) { \
    return Node(*rhs.device, ir::Builder::op_name(Node(*rhs.device, rhs.get_element_type(), lhs).get_value(), rhs.value)); \
  }\
  friend Node def_name(Node lhs, tensor::Constant rhs) { \
    return Node(*lhs.device, ir::Builder::op_name(lhs.value, Node(*lhs.device, lhs.get_element_type(), rhs).get_value())); \
  }

  #define BINOP(def_name, op_name) \
  Node def_name(Node rhs) {\
    return Node(*device, ir::Builder::op_name(value, rhs.value));\
  }\
  Node def_name(tensor::Constant rhs) {\
    return Node(*device, ir::Builder::op_name(value, Node(*device, get_element_type(), rhs).get_value()));\
  }

  #define UNIOP(def_name, op_name) \
  Node def_name() {\
    return Node(*device, ir::Builder::op_name(value));\
  }

  #define REDUCEOP(def_name, op_name) \
  Node def_name(const std::vector<int>& axis = tensor::AllAxis) {\
    return Node(*device, ir::Builder::op_name(value, ir::IntListValue(unwrap_axis(axis, value))));\
  }

  #include "operators.inc"

  Node operator-() {
    return Node(*device, ir::Builder::Neg(value));
  }

  friend std::ostream& operator<<(std::ostream& os, const Node& node) {
    return os << node.value;
  }
  ir::Value get_value() const { return value; }
  tensor::DataType get_datatype() const { return value.get_datatype(); }
  tensor::ElementType get_element_type() const { return value.get_datatype().get_element_type(); }
  tensor::Shape get_shape() const { return value.get_datatype().get_shape(); }
private:
  GraphDevice* device { nullptr };
  ir::Value value;
};

class Var final {
public:
  Var() = default;
  Var(GraphDevice& device, Tensor&& tensor, bool nograd=false) : 
    device(&device), var(ir::Var::create(std::move(tensor), nograd)) { }

  operator Node() {
    return Node(*device, ir::Value(var));
  }

  void set_tensor(Tensor&& tensor) {
    var->set_tensor(std::move(tensor));
  }

  const Tensor& get_tensor() const {
    return var->get_tensor();
  }

  const Tensor& get_grad() const {
    return var->get_grad();
  }
private:
  GraphDevice* device{nullptr};
  ir::VarPtr var;
};

class PlaceHolder final {
public:
  PlaceHolder() = default;
  PlaceHolder(GraphDevice& device, tensor::DataType datatype) : 
    device(&device), var(ir::Var::create(datatype, true)) { }

  operator Node() {
    return Node(*device, ir::Value(var));
  }
  void set_tensor(Tensor&& tensor) {
    var->set_tensor(std::move(tensor));
  }
  const Tensor& get_tensor() const {
    return var->get_tensor();
  }
private:
  GraphDevice* device { nullptr };
  ir::VarPtr var;
};

class GraphDevice {
public:
  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  template<signed_type... Sz> \
  Node zeros_##name(Sz... sz) { return Node(*this, backend().zeros_##name(sz...)); } \
  template<signed_type... Sz> \
  Var zeros_var_##name(Sz... sz) { return Var(*this, backend().zeros_##name(sz...)); } \
  template<signed_type... Sz> \
  Node ones_##name(Sz... sz) { return Node(*this, backend().ones_##name(sz...)); } \
  template<signed_type... Sz> \
  Var ones_var_##name(Sz... sz) { return Var(*this, backend().ones_##name(sz...)); } \
  template<signed_type... Sz> \
  Node uniform_##name(Sz... sz) { return Node(*this, backend().uniform_##name(sz...)); } \
  template<signed_type... Sz> \
  Var uniform_var_##name(Sz... sz) { return Var(*this, backend().uniform_##name(sz...)); } \
  template<signed_type... Sz> \
  Node norm_##name(Sz... sz) { return Node(*this, backend().normalize(backend().uniform_##name(sz...) - 0.5)); } \
  template<signed_type... Sz> \
  Var norm_var_##name(Sz... sz) { return Var(*this, backend().normalize(backend().uniform_##name(sz...) - 0.5)); } \
  template<signed_type... Sz> \
  PlaceHolder placeholder_##name(Sz... sz) { return PlaceHolder(*this, tensor::DataType(tensor::ElementType::enum_name, tensor::Shape({sz...}))); }
  #include <katoml/mltensor/element_type.inc>
  #undef ELEMENT_TYPE

  template<typename T>
  Node tensor(std::initializer_list<T> data) {
    return Node(*this, backend().tensor<T>(data));
  }

  template<typename T>
  Node tensor(std::initializer_list<std::initializer_list<T>> data) {
    return Node(*this, backend().tensor<T>(data));
  }

  template<typename T>
  Node tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> data) {
    return Node(*this, backend().tensor<T>(data));
  }

  template<typename T>
  Node from_vector(const std::vector<T>& data) {
    return Node(*this, backend().from_vector<T>(data));
  }

  template<typename T>
  Node from_vector(const std::vector<std::vector<T>>& data) {
    return Node(*this, backend().from_vector<T>(data));
  }

  template<typename T>
  Node from_vector(const std::vector<std::vector<std::vector<T>>>& data) {
    return Node(*this, backend().from_vector<T>(data));
  }

  Node constant(Tensor&& tensor) {return Node(*this, std::move(tensor)); }
  Node constant(tensor::ElementType element_type, tensor::Constant val) { return Node(*this, backend().constant(element_type, val)); }
  Var var(Tensor&& tensor) {return Var(*this, std::move(tensor)); }
  PlaceHolder placeholder(tensor::DataType datatype) {return PlaceHolder(*this, datatype); }

  #define BINOP_CPP(def_name, op_name) ;

  #define BINOP(def_name, op_name) \
  Node def_name(Node lhs, Node rhs) {\
    return lhs.def_name(rhs);\
  }\
  Node def_name(Node lhs, tensor::Constant rhs) {\
    return lhs.def_name(rhs);\
  }\
  Node def_name(tensor::Constant lhs, Node rhs) {\
    return constant(rhs.get_element_type(), lhs).def_name(rhs);\
  }

  #define UNIOP(def_name, op_name) \
  Node def_name(Node val) {\
    return val.def_name();\
  }

  #define REDUCEOP(def_name, op_name) \
  Node def_name(Node val, const std::vector<int>& axis = tensor::AllAxis) {\
    return val.def_name(axis);\
  }

  #include "operators.inc"

  tensor::Backend& backend() { return *backend_; }
protected:
  GraphDevice(std::unique_ptr<tensor::Backend>&& backend) 
    : backend_(std::move(backend)) {}
  
  std::unique_ptr<tensor::Backend> backend_;
};

Node::Node(GraphDevice& device, tensor::ElementType element_type, tensor::Constant val) : 
  Node(device, device.backend().constant(element_type, val)){
}

}
}
