#pragma once

#include "katoml/mlcompiler/tensor.hpp"
#include "types.hpp"

namespace katoml {
namespace compiler {
namespace ir {

class Var;

using VarPtr = std::shared_ptr<Var>;

class Var {
public:
  Var(tensor::DataType datatype, bool nograd = false) : 
    datatype(datatype), nograd(nograd) {}

  Var(TensorPtr&& tensor, bool nograd = false) : 
    tensor(std::move(tensor)), datatype(this->tensor->get_datatype()), nograd(nograd) {}

  inline static VarPtr create(Tensor&& tensor, bool nograd = false) {
    return std::make_shared<Var>(std::make_shared<Tensor>(std::move(tensor)), nograd);
  }
  inline static VarPtr create(tensor::DataType datatype, bool nograd = false) {
    return std::make_shared<Var>(datatype, nograd);
  }
  const tensor::DataType get_datatype() const {
    return datatype;
  }

  bool has_tensor() const {
    return tensor != nullptr;
  }

  const Tensor& get_tensor() const {
    return *tensor;
  }
  TensorPtr get_tensor_ptr() const {
    return tensor;
  }
  void set_tensor(Tensor&& tensor) {
    this->tensor = std::make_shared<Tensor>(std::move(tensor));
  }
  const Tensor& get_grad() const {
    return *grad;
  }
  TensorPtr get_grad_ptr() const {
    return grad;
  }
  void clear_grad() {
    this->grad = std::make_shared<Tensor>(get_tensor().get_backed().zeros(get_tensor().get_datatype()));
  }
  void add_grad(const Tensor& grad) {
    assert(!nograd);
    (*this->grad) += grad;
  }
  bool is_nograd() const { return nograd; }

  Var(const Var&) = delete;
  Var& operator=(const Var&) = delete;
  Var(Var&&) = default;
  Var& operator=(Var&&) = default;
private:
  TensorPtr tensor;
  TensorPtr grad;
  tensor::DataType datatype;
  bool nograd;
};


}
}
}