#pragma once

#include "katoml/mlcompiler/tensor.hpp"
#include "types.hpp"

namespace katoml {
namespace compiler {
namespace ir {

template<class Backend>
class Var;

template<class Backend>
using VarPtr = std::shared_ptr<Var<Backend>>;

template<class Backend>
class Var {
public:
  Var(tensor::DataType datatype, bool nograd = false) : 
    datatype(datatype), nograd(nograd) {}

  Var(TTensorPtr&& tensor, bool nograd = false) : 
    tensor(std::move(tensor)), datatype(this->tensor->get_datatype()), nograd(nograd) {}

  inline static VarPtr<Backend> create(TTensor&& tensor, bool nograd = false) {
    return std::make_shared<Var<Backend>>(std::make_shared<TTensor>(std::move(tensor)), nograd);
  }
  inline static VarPtr<Backend> create(tensor::DataType datatype, bool nograd = false) {
    return std::make_shared<Var<Backend>>(datatype, nograd);
  }
  const tensor::DataType get_datatype() const {
    return datatype;
  }

  const TTensor& get_tensor() const {
    return *tensor;
  }
  TTensorPtr get_tensor_ptr() const {
    return tensor;
  }
  void set_tensor(TTensor&& tensor) {
    this->tensor = std::make_shared<TTensor>(std::move(tensor));
  }
  const TTensor& get_grad() const {
    return *grad;
  }
  TTensorPtr get_grad_ptr() const {
    return grad;
  }
  void set_grad(TTensor&& grad) {
    assert(!nograd);
    this->grad = std::make_shared<TTensor>(std::move(grad));
  }
  bool is_nograd() const { return nograd; }

  Var(const Var&) = delete;
  Var& operator=(const Var&) = delete;
  Var(Var&&) = default;
  Var& operator=(Var&&) = default;
private:
  TTensorPtr tensor;
  TTensorPtr grad;
  tensor::DataType datatype;
  bool nograd;
};


}
}
}