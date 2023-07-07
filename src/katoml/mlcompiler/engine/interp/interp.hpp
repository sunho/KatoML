#pragma once

#include "../engine.hpp"
#include "katoml/mlcompiler/ir/value.hpp"
#include "katoml/mlcompiler/tensor.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/types.hpp"
#include <functional>
#include <memory>

namespace katoml {
namespace compiler {
namespace engine {

class InterpComputeGraph {
public:
  bool exists(const ir::Node& node) {
    return cache.find(node.get_id()) != cache.end();
  }
  TensorPtr get(const ir::Node& node) {
    return cache.at(node.get_id());
  }
  TensorPtr set(const ir::Node& node, TensorPtr tensor) {
    return cache[node.get_id()] = tensor;
  }
private:
  std::map<uint64_t, TensorPtr> cache;
};

class ForwardEvalVisitor {
public:
  ForwardEvalVisitor(InterpComputeGraph& cg) : cg(cg) {}
  const Tensor& evaluate(ir::Value value) {
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
      return *cg.set(node, ir::NodeVisitorCaller<TensorPtr, ForwardEvalVisitor>().call(*this, node));
    }
    case ir::ValueType::Tensor: 
      return *value.as_tensor();
    case ir::ValueType::Var: 
      return value.as_var()->get_tensor();
    }
  }
  #define DECL_NODE(OP, ARGS, PARAMS, TYPES) inline TensorPtr OP(ARGS);
  #include "../../ir/ir.inc"
  #undef DECL_NODE

private:
  TensorPtr wrap(Tensor&& tensor) {
    return std::make_shared<Tensor>(std::move(tensor));
  }

  InterpComputeGraph& cg;
};

class BackwardVisitor {
public:
  struct Diff {
    const Tensor& dRdY;
    const Tensor& cur;
  };

  BackwardVisitor(InterpComputeGraph& cg, tensor::Backend& backend) : cg(cg), backend(backend) {}
  void backward(ir::Value value, const Tensor& cur_dRdY) {
    switch (value.get_type()) {
    case ir::ValueType::IntList:
    case ir::ValueType::Void:
    case ir::ValueType::Str:
      assert(false);
      return;
    case ir::ValueType::Node: {
      Diff diff{cur_dRdY, *cg.get(*value.as_node())};
      ir::NodeVisitorCallerWithUserData<void, Diff, BackwardVisitor>().call(*this, *value.as_node(), diff);
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
  const Tensor& evaluate(ir::Value value) {
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
  inline Tensor reduce_sum_impl(Diff& diff, const Tensor& val, ir::IntListValue axis);

  tensor::Backend& backend;
  InterpComputeGraph& cg;
};

class InterpEngine : public Engine {
public:
  InterpEngine(tensor::Backend& backend) : Engine(backend) {}
  friend class ProgramImpl;
  std::unique_ptr<Program> compile(ir::Value value) {
    return std::make_unique<ProgramImpl>(*this, value);
  }
private:
  using ComputeGraph = InterpComputeGraph;
  class ProgramImpl : public Program {
  public:
    ProgramImpl(InterpEngine& parent, ir::Value value) : parent(parent), value(value) {}
    Tensor forward() override {
      ForwardEvalVisitor visitor(cg);
      return visitor.evaluate(value).copy();
    }
    bool backward() override {
      BackwardVisitor visitor(cg, parent.get().backend);
      visitor.backward(value, parent.get().backend.ones(value.get_datatype()));
      return true;
    }
    void reset() override {
      cg = ComputeGraph();
    }
  private:
    ir::Value get_value() const {
      return value;
    }
    ir::Value value;
    ComputeGraph cg;
    std::reference_wrapper<InterpEngine> parent;
  };
};

TensorPtr ForwardEvalVisitor::Add(ir::Value lhs, ir::Value rhs) {
  return wrap(evaluate(lhs) + evaluate(rhs));
}

TensorPtr ForwardEvalVisitor::Sub(ir::Value lhs, ir::Value rhs) {
  return wrap( evaluate(lhs) - evaluate(rhs));
}

TensorPtr ForwardEvalVisitor::Mul(ir::Value lhs, ir::Value rhs) {
  return wrap(evaluate(lhs) * evaluate(rhs));
}

TensorPtr ForwardEvalVisitor::Max(ir::Value lhs, ir::Value rhs) {
  return wrap(evaluate(lhs).max(evaluate(rhs)));
}

TensorPtr ForwardEvalVisitor::Min(ir::Value lhs, ir::Value rhs) {
  return wrap(evaluate(lhs).min(evaluate(rhs)));
}

TensorPtr ForwardEvalVisitor::Div(ir::Value lhs, ir::Value rhs) {
  return wrap(evaluate(lhs) / evaluate(rhs));
}

static inline Tensor safe_log(const Tensor& val) {
  const tensor::Constant LOG_MIN(1e-15f);
  return val.clip(LOG_MIN, tensor::Constant::max()).log();
}

TensorPtr ForwardEvalVisitor::SoftMax(ir::Value val) {
  const auto& val_ = evaluate(val);
  auto max_ = val_.reduce_max({-1});
  if (val_.get_ndims() > 1) max_.extend_axis();
  auto z = val_ - max_;
  auto exp_ = z.exp();
  auto sum_ = exp_.sum({-1});
  if (val_.get_ndims() > 1) sum_.extend_axis();
  return wrap(exp_ / sum_);
}

TensorPtr ForwardEvalVisitor::LogSoftMax(ir::Value val) {
  const auto& val_ = evaluate(val);
  auto mx = val_.reduce_max({-1});
  if (val_.get_ndims() > 1) mx.extend_axis();
  auto z = val_ - mx;
  auto exp = z.exp();
  auto sum = exp.sum({-1});
  if (val_.get_ndims() > 1) sum.extend_axis();
  return wrap(z - safe_log(sum));
}

TensorPtr ForwardEvalVisitor::Log(ir::Value val) {
  return wrap(safe_log( evaluate(val)));
}

TensorPtr ForwardEvalVisitor::Neg(ir::Value val) {
  return wrap(-evaluate(val));
}

TensorPtr ForwardEvalVisitor::ReduceSum(ir::Value val, ir::IntListValue axis) {
  return wrap(evaluate(val).sum(axis.as_int_list()));
}

TensorPtr ForwardEvalVisitor::ReduceMean(ir::Value val, ir::IntListValue axis) {
  return wrap(evaluate(val).mean(axis.as_int_list()));
}

TensorPtr ForwardEvalVisitor::MatMul(ir::Value lhs, ir::Value rhs) {
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

void BackwardVisitor::Add(Diff& diff, ir::Value lhs, ir::Value rhs) {
  auto process = [&](const Tensor& val, const Tensor& other) {
    return diff.dRdY.sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

void BackwardVisitor::Sub(Diff& diff, ir::Value lhs, ir::Value rhs) {
  auto process = [&](const Tensor& val, const Tensor& other) {
    return diff.dRdY.sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, -process(evaluate(rhs), evaluate(lhs)));
}

void BackwardVisitor::Max(Diff& diff, ir::Value lhs, ir::Value rhs) {
  auto process = [&](const Tensor& val, const Tensor& other) {
    return ((val >= other) * diff.dRdY).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

void BackwardVisitor::Min(Diff& diff, ir::Value lhs, ir::Value rhs) {
  auto process = [&](const Tensor& val, const Tensor& other) {
    return ((val <= other) * diff.dRdY).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

void BackwardVisitor::Mul(Diff& diff, ir::Value lhs, ir::Value rhs) {
  auto process = [&](const Tensor& val, const Tensor& other) {
    return (diff.dRdY * other).sum(find_broadcast_axis(val.get_shape(), other.get_shape()));
  };
  backward(lhs, process(evaluate(lhs), evaluate(rhs)));
  backward(rhs, process(evaluate(rhs), evaluate(lhs)));
}

void BackwardVisitor::Div(Diff& diff, ir::Value lhs, ir::Value rhs) {
  assert(false);
}

void BackwardVisitor::SoftMax(Diff& diff, ir::Value val) {

}

void BackwardVisitor::LogSoftMax(Diff& diff, ir::Value val) {
  const auto& val_ = evaluate(val);
  auto sum = diff.dRdY.sum({-1});
  if (val_.get_ndims() > 1) sum.extend_axis();
  backward(val, -sum *  val_.exp() + diff.dRdY);
}

void BackwardVisitor::Log(Diff& diff, ir::Value val) {
}

void BackwardVisitor::Neg(Diff& diff, ir::Value val) {
  backward(val, -diff.dRdY);
}

Tensor BackwardVisitor::reduce_sum_impl(Diff& diff, const Tensor& val, ir::IntListValue axis) {
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

void BackwardVisitor::ReduceSum(Diff& diff, ir::Value val, ir::IntListValue axis) {
  backward(val, reduce_sum_impl(diff, evaluate(val), axis));
}

void BackwardVisitor::ReduceMean(Diff& diff, ir::Value val, ir::IntListValue axis) {
  const auto& val_ = evaluate(val);
  int64_t total = tensor::calculate_reduced_count(val_.get_shape(), axis.as_int_list());
  backward(val, reduce_sum_impl(diff, val_, axis) / backend.constant(val_.get_element_type(), total));
}

void BackwardVisitor::MatMul(Diff& diff, ir::Value lhs, ir::Value rhs) {
  backward(lhs, diff.dRdY.matmul(evaluate(rhs).transposed()));
  backward(rhs, evaluate(lhs).transposed().matmul(diff.dRdY));
}

}
}
}