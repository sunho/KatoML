#pragma once

#include "../ir/ir.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/types.hpp"

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
  Node(GraphDevice& device, ir::Value value) : 
    device(device), value(value) {}
  Node(GraphDevice& device, Tensor&& tensor) :
    device(device), value(std::make_shared<Tensor>(std::move(tensor))) {}
  friend Node operator+(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder::Add(lhs.value, rhs.value));
  }
  friend Node operator-(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder::Sub(lhs.value, rhs.value));
  }
  friend Node operator*(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder::Mul(lhs.value, rhs.value));
  }
  friend Node operator/(Node lhs, Node rhs) {
    return Node(lhs.device, ir::Builder::Div(lhs.value, rhs.value));
  }
  Node max(Node rhs) {
    return Node(device, ir::Builder::Max(value, rhs.value));
  }
  Node min(Node rhs) {
    return Node(device, ir::Builder::Min(value, rhs.value));
  }
  Node operator-() {
    return Node(device, ir::Builder::Neg(value));
  }
  Node matmul(Node val) const {
    return Node(device, ir::Builder::MatMul(value, val.value));
  }
  Node log() const {
    return Node(device, ir::Builder::Log(value));
  }
  Node sum(const std::vector<int>& axis = tensor::AllAxis) {
    return Node(device, ir::Builder::ReduceSum(value, ir::IntListValue(unwrap_axis(axis, value))));
  }
  Node mean(const std::vector<int>& axis = tensor::AllAxis) {
    return Node(device, ir::Builder::ReduceMean(value, ir::IntListValue(unwrap_axis(axis, value))));
  }
  Node softmax() const {
    return Node(device, ir::Builder::SoftMax(value));
  }
  Node log_softmax() const {
    return Node(device, ir::Builder::LogSoftMax(value));
  }
  friend std::ostream& operator<<(std::ostream& os, const Node& node) {
    return os << node.value;
  }
  ir::Value get_value() const { return value; }
private:
  GraphDevice& device;
  ir::Value value;
};

class Var final {
public:
  Var(GraphDevice& device, Tensor&& tensor, bool nograd=false) : 
    device(device), var(ir::Var::create(std::move(tensor), nograd)) { }

  operator Node() {
    return Node(device, ir::Value(var));
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
  GraphDevice& device;
  ir::VarPtr var;
};

class PlaceHolder final {
public:
  PlaceHolder(GraphDevice& device, tensor::DataType datatype) : 
    device(device), var(ir::Var::create(datatype, true)) { }
  
  operator Node() {
    return Node(device, ir::Value(var));
  }
  void set_tensor(Tensor&& tensor) {
    var->set_tensor(std::move(tensor));
  }
  const Tensor& get_tensor() const {
    return var->get_tensor();
  }
private:
  GraphDevice& device;
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
  Node tensor(const std::vector<T>& data) {
    return Node(*this, backend().template tensor<T>(data));
  }

  template<typename T>
  Node tensor(const std::vector<std::vector<T>>& data) {
    return Node(*this, backend().template tensor<T>(data));
  }

  template<typename T>
  Node tensor(const std::vector<std::vector<std::vector<T>>>& data) {
    return Node(*this, backend().template tensor<T>(data));
  }

  Node constant(Tensor&& tensor) {return Node(*this, std::move(tensor)); }
  Var var(Tensor&& tensor) {return Var(*this, std::move(tensor)); }
  PlaceHolder placeholder(tensor::DataType datatype) {return PlaceHolder(*this, datatype); }
  
  Node max(Node a, Node b) { return a.max(b); }
  Node min(Node a, Node b) { return a.min(b); }
  Node matmul(Node a, Node b) { return a.matmul(b); }
  Node log(Node a) { return a.log(); }
  Node softmax(Node a) { return a.softmax(); }
  Node log_softmax(Node a) { return a.log_softmax(); }
  Node sum(Node a, const std::vector<int>& axis = tensor::AllAxis) { return a.sum(axis); }
  Node mean(Node a, const std::vector<int>& axis = tensor::AllAxis) { return a.mean(axis); }

  tensor::Backend& backend() { return *backend_; }
protected:
  GraphDevice(std::unique_ptr<tensor::Backend>&& backend) 
    : backend_(std::move(backend)) {}
  
  std::unique_ptr<tensor::Backend> backend_;
};

}
}
