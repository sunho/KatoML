#pragma once

#include "../ir/ir.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/types.hpp"

namespace katoml {
namespace compiler {

template <typename T> concept signed_type = std::is_signed_v<T>; 

template<class Engine, class Executor>
class IDevice;

template<typename Backend>
static inline std::vector<int> unwrap_axis(const std::vector<int>& axis, ir::Value<Backend> value) {
  if (axis == tensor::AllAxis) {
    std::vector<int> axis_(value.get_datatype().get_shape().get_ndims());
    std::iota(std::begin(axis_), std::end(axis_), 0);
    return axis_;
  }
  return axis;
}

template<class Device, class Backend>
class Node final {
public:
  Node(Device& device, ir::Value<Backend> value) : 
    device(device), value(value) {}
  Node(Device& device, TTensor&& tensor) :
    device(device), value(std::make_shared<TTensor>(std::move(tensor))) {}
  friend Node operator+(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder<Backend>::Add(lhs.value, rhs.value));
  }
  friend Node operator-(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder<Backend>::Sub(lhs.value, rhs.value));
  }
  friend Node operator*(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder<Backend>::Mul(lhs.value, rhs.value));
  }
  friend Node operator/(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder<Backend>::Div(lhs.value, rhs.value));
  }
  Node max(Node rhs) {
    return Node(device, ir::Builder<Backend>::Max(value, rhs.value));
  }
  Node min(Node rhs) {
    return Node(device, ir::Builder<Backend>::Min(value, rhs.value));
  }
  Node operator-() {
    return Node(device, ir::Builder<Backend>::Neg(value));
  }
  Node matmul(Node val) const {
    return Node(device, ir::Builder<Backend>::MatMul(value, val.value));
  }
  Node log() const {
    return Node(device, ir::Builder<Backend>::Log(value));
  }
  Node sum(const std::vector<int>& axis = tensor::AllAxis) {
    return Node(device, ir::Builder<Backend>::ReduceSum(value, ir::IntListValue<Backend>(unwrap_axis(axis, value))));
  }
  Node mean(const std::vector<int>& axis = tensor::AllAxis) {
    return Node(device, ir::Builder<Backend>::ReduceMean(value, ir::IntListValue<Backend>(unwrap_axis(axis, value))));
  }
  Node softmax() const {
    return Node(device, ir::Builder<Backend>::SoftMax(value));
  }
  Node log_softmax() const {
    return Node(device, ir::Builder<Backend>::LogSoftMax(value));
  }
  friend std::ostream& operator<<(std::ostream& os, const Node& node) {
    return os << node.value;
  }
  ir::Value<Backend> get_value() const { return value; }
private:
  Device& device;
  ir::Value<Backend> value;
};

template<class Device, class Backend>
class Var final {
public:
  Var(Device& device, TTensor&& tensor, bool nograd=false) : 
    device(device), var(ir::Var<Backend>::create(std::move(tensor), nograd)) { }

  operator Node<Device, Backend>() {
    return Node<Device, Backend>(device, ir::Value(var));
  }

  void set_tensor(TTensor&& tensor) {
    var->set_tensor(std::move(tensor));
  }

  const TTensor& get_tensor() const {
    return var->get_tensor();
  }

  const TTensor& get_grad() const {
    return var->get_grad();
  }
private:
  Device& device;
  ir::VarPtr<Backend> var;
};

template<class Device, class Backend>
class PlaceHolder final {
public:
  PlaceHolder(Device& device, tensor::DataType datatype) : 
    device(device), var(ir::Var<Backend>::create(datatype, true)) { }
  operator Node<Device, Backend>() {
    return Node<Device, Backend>(device, ir::Value(var));
  }
  void set_tensor(TTensor&& tensor) {
    var->set_tensor(std::move(tensor));
  }
  const TTensor& get_tensor() const {
    return var->get_tensor();
  }
private:
  Device& device;
  ir::VarPtr<Backend> var;
};

template<class Device, class Backend>
class GraphBuilder {
public:
  GraphBuilder(Device& device) 
    : device(device) {}
  using Node = Node<Device, Backend>;
  using Var = Var<Device, Backend>;
  using PlaceHolder = PlaceHolder<Device, Backend>;

  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  template<signed_type... Sz> \
  Node zeros_##name(Sz... sz) { return Node(device, device.zeros_##name(sz...)); } \
  template<signed_type... Sz> \
  Var zeros_var_##name(Sz... sz) { return Var(device, device.zeros_##name(sz...)); } \
  template<signed_type... Sz> \
  Node ones_##name(Sz... sz) { return Node(device, device.ones_##name(sz...)); } \
  template<signed_type... Sz> \
  Var ones_var_##name(Sz... sz) { return Var(device, device.ones_##name(sz...)); } \
  template<signed_type... Sz> \
  Node uniform_##name(Sz... sz) { return Node(device, device.uniform_##name(sz...)); } \
  template<signed_type... Sz> \
  Var uniform_var_##name(Sz... sz) { return Var(device, device.uniform_##name(sz...)); } \
  template<signed_type... Sz> \
  Node norm_##name(Sz... sz) { return Node(device, device.normalize(device.uniform_##name(sz...) - device.constant(0.5f))); } \
  template<signed_type... Sz> \
  Var norm_var_##name(Sz... sz) { return Var(device, device.normalize(device.uniform_##name(sz...) - device.constant(0.5f))); } \
  template<signed_type... Sz> \
  PlaceHolder placeholder_##name(Sz... sz) { return PlaceHolder(device, tensor::DataType(tensor::ElementType::enum_name, tensor::Shape({sz...}))); }
  #include <katoml/mltensor/element_type.inc>
  #undef ELEMENT_TYPE

  template<typename T>
  Node tensor(const std::vector<T>& data) {
    return Node(device, device.template tensor<T>(data));
  }

  template<typename T>
  Node tensor(const std::vector<std::vector<T>>& data) {
    return Node(device, device.template tensor<T>(data));
  }

  template<typename T>
  Node tensor(const std::vector<std::vector<std::vector<T>>>& data) {
    return Node(device, device.template tensor<T>(data));
  }

  Node constant(TTensor&& tensor) {return Node(device, std::move(tensor)); }
  Var var(TTensor&& tensor) {return Var(device, std::move(tensor)); }
  PlaceHolder placeholder(TTensor&& tensor) {return PlaceHolder(device, std::move(tensor)); }
  
  Node max(Node a, Node b) { return a.max(b); }
  Node min(Node a, Node b) { return a.min(b); }
  Node matmul(Node a, Node b) { return a.matmul(b); }
  Node log(Node a) { return a.log(); }
  Node softmax(Node a) { return a.softmax(); }
  Node log_softmax(Node a) { return a.log_softmax(); }
  Node sum(Node a, const std::vector<int>& axis = tensor::AllAxis) { return a.sum(axis); }
  Node mean(Node a, const std::vector<int>& axis = tensor::AllAxis) { return a.mean(axis); }
private:
  Device& device;
};

}
}
