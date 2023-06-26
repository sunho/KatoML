#pragma once

#include "../engine.hpp"
#include "katoml/mlcompiler/ir/value.hpp"
#include "katoml/mlcompiler/tensor.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/types.hpp"
#include <functional>

namespace katoml {
namespace compiler {
namespace engine {

template<typename Backend>
class InterpComputeGraph {
public:
  bool exists(const ir::Node<Backend>& node) {
    return cache.find(node.get_id()) != cache.end();
  }
  TTensorPtr get(const ir::Node<Backend>& node) {
    return cache.at(node.get_id());
  }
  TTensorPtr set(const ir::Node<Backend>& node, TTensorPtr tensor) {
    return cache[node.get_id()] = tensor;
  }
private:
  std::map<uint64_t, TTensorPtr> cache;
};

template<class Backend>
class ForwardEvalVisitor {
public:
  ForwardEvalVisitor(InterpComputeGraph<Backend>& cg) : cg(cg) {}
  const TTensor& evaluate(ir::Value<Backend> value) {
    switch (value.get_type()) {
    case ir::ValueType::IntList:
    case ir::ValueType::Void:
    case ir::ValueType::Str:
      assert(false);
    case ir::ValueType::Node: {
      const auto& node = *value.as_node();
      if (cg.exists(node)) {
        return *cg.get(node);
      }
      return *cg.set(node, ir::NodeVisitorCaller<Backend, TTensorPtr, ForwardEvalVisitor<Backend>>().call(*this, node));
    }
    case ir::ValueType::Tensor: 
      return *value.as_tensor();
    case ir::ValueType::Var: 
      return value.as_var()->get_tensor();
    }
  }
  #define DECL_NODE(OP, ARGS, PARAMS, TYPES) inline TTensorPtr OP(ARGS);
  #include "../../ir/ir.inc"
  #undef DECL_NODE

private:
  TTensorPtr wrap(TTensor&& tensor) {
    return std::make_shared<TTensor>(std::move(tensor));
  }

  InterpComputeGraph<Backend>& cg;
};

template<class Backend>
class BackwardVisitor {
public:
  struct Diff {
    const TTensor& dRdY;
    const TTensor& cur;
  };

  BackwardVisitor(InterpComputeGraph<Backend>& cg, Backend& backend) : cg(cg), backend(backend) {}
  void backward(ir::Value<Backend> value, const TTensor& cur_dRdY) {
    switch (value.get_type()) {
    case ir::ValueType::IntList:
    case ir::ValueType::Void:
    case ir::ValueType::Str:
      assert(false);
      return;
    case ir::ValueType::Node: {
      Diff diff{cur_dRdY, *cg.get(*value.as_node())};
      ir::NodeVisitorCallerWithUserData<Backend, void, Diff, BackwardVisitor<Backend>>().call(*this, *value.as_node(), diff);
      return;
    }
    case ir::ValueType::Tensor:
      return;
    case ir::ValueType::Var:
      if(!value.as_var()->is_nograd())
        value.as_var()->set_grad(cur_dRdY.copy());
      return;
    }
  }
  const TTensor& evaluate(ir::Value<Backend> value) {
    switch (value.get_type()) {
    case ir::ValueType::IntList:
    case ir::ValueType::Void:
    case ir::ValueType::Str:
      assert(false);
    case ir::ValueType::Node: {
      const auto& node = *value.as_node();
      assert(cg.exists(node));
      return *cg.get(node);
    }
    case ir::ValueType::Tensor: 
      return *value.as_tensor();
    case ir::ValueType::Var: 
      return value.as_var()->get_tensor();
    }
  }
  #define DECL_NODE(OP, ARGS, PARAMS, TYPES) inline void OP(Diff& diff, ARGS);
  #include "../../ir/ir.inc"
  #undef DECL_NODE

private:
  TTensor reduce_sum_impl(Diff& diff, const TTensor& val, ir::IntListValue<Backend> axis);

  Backend& backend;
  InterpComputeGraph<Backend>& cg;
};

template<typename Backend>
class InterpEngineImpl {
public:
  InterpEngineImpl(Backend& backend) :
   backend(backend) {}
  using ComputeGraph = InterpComputeGraph<Backend>;
  class Program {
  public:
    friend class InterpEngineImpl;
    Program(InterpEngineImpl& parent, ir::Value<Backend> value) : parent(parent), value(value) {}
    TTensor forward() {
      ForwardEvalVisitor visitor(cg);
      return visitor.evaluate(value).copy();
    }
    bool backward() {
      BackwardVisitor visitor(cg, parent.get().backend);
      visitor.backward(value, parent.get().backend.ones(value.get_datatype()));
      return true;
    }
    void reset() {
      cg = ComputeGraph();
    }
  private:
    ir::Value<Backend> get_value() const {
      return value;
    }
    ir::Value<Backend> value;
    ComputeGraph cg;
    std::reference_wrapper<InterpEngineImpl> parent;
  };
  friend class Program;

  Program compile(ir::Value<Backend> value) {
    return Program(*this, value);
  }
private:
  Backend& backend;
};

template<typename Backend>
using InterpEngine = Engine<Backend, InterpEngineImpl<Backend>>;

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Add(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs) + evaluate(rhs));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Sub(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap( evaluate(lhs) - evaluate(rhs));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Mul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs) * evaluate(rhs));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Max(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs).max(evaluate(rhs)));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Min(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs).min(evaluate(rhs)));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Div(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs) / evaluate(rhs));
}

template<class Backend>
static inline TTensor safe_log(const TTensor& val) {
  const tensor::Constant LOG_MIN(1e-15f);
  return val.clip(LOG_MIN, tensor::Constant::max()).log();
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::SoftMax(ir::Value<Backend> val) {
  const auto& val_ = evaluate(val);
  auto max_ = val_.reduce_max({-1});
  if (val_.get_ndims() > 1) max_.extend_axis();
  auto z = val_ - max_;
  auto exp_ = z.exp();
  auto sum_ = exp_.sum({-1});
  if (val_.get_ndims() > 1) sum_.extend_axis();
  return wrap(exp_ / sum_);
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::LogSoftMax(ir::Value<Backend> val) {
  const auto& val_ = evaluate(val);
  auto mx = val_.reduce_max({-1});
  if (val_.get_ndims() > 1) mx.extend_axis();
  auto z = val_ - mx;
  auto exp = z.exp();
  auto sum = exp.sum({-1});
  if (val_.get_ndims() > 1) sum.extend_axis();
  return wrap(z - safe_log<Backend>(sum));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Log(ir::Value<Backend> val) {
  return wrap(safe_log<Backend>( evaluate(val)));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::Neg(ir::Value<Backend> val) {
  return wrap(-evaluate(val));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::ReduceSum(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  return wrap(evaluate(val).sum(axis.as_int_list()));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::ReduceMean(ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  return wrap(evaluate(val).mean(axis.as_int_list()));
}

template<class Backend>
TTensorPtr ForwardEvalVisitor<Backend>::MatMul(ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  return wrap(evaluate(lhs).matmul(evaluate(rhs)));
}

static inline std::vector<int> find_broadcast_axis(tensor::Shape vshape, tensor::Shape oshape) {
  int min_dim = std::min(vshape.get_ndims(), oshape.get_ndims());
  int max_dim = std::max(vshape.get_ndims(), oshape.get_ndims());
  std::vector<int> axis;
  for (int i=0;i<min_dim;i++){
    if (vshape[-i-1] == 1 && oshape[-i-1] != 1) {
      axis.push_back(max_dim-i-1);
    }
  }
  if (max_dim > vshape.get_ndims()) {
    for (int i=min_dim;i<max_dim;i++){
      axis.push_back(max_dim-i-1);
    }
  }
  return axis;
}

template<class Backend>
void BackwardVisitor<Backend>::Add(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  auto process = [&](const TTensor& val, const TTensor& other) {
    return diff.dRdY.sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

template<class Backend>
void BackwardVisitor<Backend>::Sub(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  auto process = [&](const TTensor& val, const TTensor& other) {
    return diff.dRdY.sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, -process(evaluate(rhs), evaluate(lhs)));
}

template<class Backend>
void BackwardVisitor<Backend>::Max(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  auto process = [&](const TTensor& val, const TTensor& other) {
    return ((val >= other) * diff.dRdY).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

template<class Backend>
void BackwardVisitor<Backend>::Min(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  auto process = [&](const TTensor& val, const TTensor& other) {
    return ((val <= other) * diff.dRdY).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

template<class Backend>
void BackwardVisitor<Backend>::Mul(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  auto process = [&](const TTensor& val, const TTensor& other) {
    return (diff.dRdY * other).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

template<class Backend>
void BackwardVisitor<Backend>::Div(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  assert(false);
}

template<class Backend>
void BackwardVisitor<Backend>::SoftMax(Diff& diff, ir::Value<Backend> val) {

}

template<class Backend>
void BackwardVisitor<Backend>::LogSoftMax(Diff& diff, ir::Value<Backend> val) {
  const auto& val_ = evaluate(val);
  auto sum = diff.dRdY.sum({-1});
  if (val_.get_ndims() > 1) sum.extend_axis();
  backward(val, -sum *  val_.exp() + diff.dRdY);
}

template<class Backend>
void BackwardVisitor<Backend>::Log(Diff& diff, ir::Value<Backend> val) {
}

template<class Backend>
void BackwardVisitor<Backend>::Neg(Diff& diff, ir::Value<Backend> val) {
  backward(val, -diff.dRdY);
}

template<class Backend>
TTensor BackwardVisitor<Backend>::reduce_sum_impl(Diff& diff, const TTensor& val, ir::IntListValue<Backend> axis) {
  std::vector<int> axis_ = axis.as_int_list();
  tensor::Shape shape = val.get_shape();
  std::vector<bool> ban(shape.get_ndims());
  tensor::Shape new_shape(shape.get_ndims());
  for (int i : axis_){
    ban[i] = true;
  }
  for (int i=0;i<shape.get_ndims();i++){
    if (ban[i])
      new_shape[i] = 1;
    else
      new_shape[i] = shape[i];
  }
  return backend.ones(val.get_datatype()) * diff.dRdY.reshaped(new_shape);
}

template<class Backend>
void BackwardVisitor<Backend>::ReduceSum(Diff& diff, ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  backward(val, reduce_sum_impl(diff, evaluate(val), axis));
}

template<class Backend>
void BackwardVisitor<Backend>::ReduceMean(Diff& diff, ir::Value<Backend> val, ir::IntListValue<Backend> axis) {
  const auto& val_ = evaluate(val);
  int64_t total = tensor::calculate_reduced_count(val_.get_shape(), axis.as_int_list());
  backward(val, reduce_sum_impl(diff, val_, axis) / backend.constant(val_.get_element_type(), total));
}

template<class Backend>
void BackwardVisitor<Backend>::MatMul(Diff& diff, ir::Value<Backend> lhs, ir::Value<Backend> rhs) {
  backward(lhs, diff.dRdY.matmul(evaluate(rhs).transposed()));
  backward(rhs, evaluate(lhs).transposed().matmul(diff.dRdY));
}

}
}
}