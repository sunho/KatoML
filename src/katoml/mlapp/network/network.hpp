#pragma once

#include "katoml/mlcompiler/device.hpp"
#include "katoml/mlcompiler/graph/graph.hpp"
#include "katoml/mltensor/types.hpp"
#include <algorithm>
#include <functional>
#include <memory>
#include <variant>

namespace katoml {
namespace app {
namespace network {

class Context {
public:
  Context() = default;
  Context(compiler::Device& device_) : device_(&device_) {}

  compiler::Device& device() {
    return *device_;
  }
  bool inited() {
    return device_;
  }
private:
  compiler::Device* device_ { nullptr };
};

inline thread_local Context __thread_context;

static inline void set_thread_context(Context&& context) {
  __thread_context = std::move(context);
}

class Layer;
using LayerPtr = std::shared_ptr<Layer>;

enum class LayerType {
  Compute,
  Input,
};

class Layer {
public:
  Layer(compiler::Device& device, const std::string& name, std::vector<network::LayerPtr> inputs) : 
    device(device), name(name), inputs(inputs) {}

  size_t get_num_inputs() const {
    return inputs.size();
  }

  size_t get_num_outputs() const {
    return outputs.size();
  }

  const std::string& get_name() const {
    return name;
  }

  const std::vector<LayerPtr>& ins() const {
    return inputs;
  }

  const std::vector<compiler::Node>& outs() const {
    return outputs;
  }

  void set_outs(const std::vector<compiler::Node>& outputs_) {
    this->outputs = outputs_;
  }

  const std::string& get_meta(const std::string& key) const {
    return metadata.at(key);
  }

  void set_meta(const std::string& key, const std::string& value) {
    metadata[key] = value;
  }

  LayerType get_type() const {
    return type;
  }

  void set_type(LayerType type) {
    this->type = type;
    if (type == LayerType::Input) {
      inner_data = InputLayerData{};
    }
  }

  compiler::Node out() const {
    assert(outs().size() == 1 && "ouput size is not 1");
    return outs()[0];
  }

  compiler::Var add_param(tensor::Tensor&& initial) {
    compiler::Var var = device.var(std::move(initial));
    params.push_back(var);
    return var;
  }

  const std::vector<compiler::Var>& get_params() const {
    return params;
  }

  const std::string& get_input_name() const {
    assert(type == LayerType::Input && "non input type layer does not have input name");
    return std::get<InputLayerData>(inner_data).input_name;
  }

  void set_input_name(const std::string& name) {
    assert(type == LayerType::Input && "non input type layer does not have input name");
    std::get<InputLayerData>(inner_data).input_name = name;
  }

  compiler::PlaceHolder get_input_placeholder() const {
    assert(type == LayerType::Input && "non input type layer does not have input placeholder");
    return std::get<InputLayerData>(inner_data).input_placeholder;
  }

  void set_input_placeholder(compiler::PlaceHolder placeholder) {
    assert(type == LayerType::Input && "non input type layer does not have input placeholder");
    std::get<InputLayerData>(inner_data).input_placeholder = placeholder;
  }

  compiler::Device& get_device() {
    return device;
  }

private:
  std::string name;
  LayerType type { LayerType::Compute };
  struct InputLayerData {
    std::string input_name;
    compiler::PlaceHolder input_placeholder;
  };
  std::variant<InputLayerData, std::monostate> inner_data { std::monostate() };
  compiler::Device& device;

  std::vector<network::LayerPtr> inputs;
  std::vector<compiler::Node> outputs;
  std::map<std::string, std::string> metadata;
  std::vector<compiler::Var> params;
};

template<typename F, typename... Vargs>
concept LayerDefFunc = requires(F func, compiler::Device& device, LayerPtr self, Vargs ...args) {
  { func(device, self, args...) };
};

template<typename Func>
class LayerDef {
public:
  LayerDef(const std::string& name, Func impl) : impl(impl) {}

  template<typename ...Vargs> requires LayerDefFunc<Func, Vargs...>
  LayerPtr operator()(Vargs... args) {
    assert(__thread_context.inited() && "thread context not inited");
    return (*this)(__thread_context, args...);
  }

  template<typename ...Vargs> requires LayerDefFunc<Func, Vargs...>
  LayerPtr operator()(network::Context& context, Vargs... args) {
    std::vector<LayerPtr> inputs;
    const auto add_to_input = [&](auto layer) {
      if constexpr (std::is_same<decltype(layer), LayerPtr>::value) {
        inputs.push_back(layer);
      }
    };
    (add_to_input(args),...);
    auto self = std::make_shared<Layer>(context.device(), name, inputs);
    auto outputs = impl(context.device(), self, args...);
    if constexpr (std::is_same<decltype(outputs), std::vector<compiler::Node>>::value) {
      self->set_outs(outputs);
    } else {
      self->set_outs({outputs});
    }
    return self;
  }
private:
  std::string name;
  Func impl;
};

template<typename F>
static inline LayerDef<F> layer_def(const std::string& name, F func) {
  return LayerDef<F>(name, func);
}

static inline LayerDef input = network::layer_def("Input", [](compiler::Device& device, LayerPtr self, const std::string& name, tensor::DataType datatype) {
  self->set_type(LayerType::Input);
  self->set_input_name(name);
  auto placeholder = device.placeholder(datatype);
  self->set_input_placeholder(placeholder);
  return placeholder;
});

using DenseInitializer = std::function<std::tuple<tensor::Tensor, tensor::Tensor>(compiler::Device& device, tensor::DataType input_type, size_t output_size)>;
using ActivationFunc = std::function<compiler::Node(compiler::Device& device, compiler::Node affine)>;
using LossFunc = std::function<compiler::Node(compiler::Device& device, compiler::Node predicted, compiler::Node label)>;

namespace initializer {
  static inline DenseInitializer xavier = [](compiler::Device& device, tensor::DataType input_type, int output_size) -> std::tuple<tensor::Tensor, tensor::Tensor> {
    int input_size = input_type.get_shape()[-1];
    auto W_shape = tensor::Shape({input_size, output_size});
    auto b_shape = tensor::Shape({output_size});
    auto W = device.backend().rand_normal(tensor::DataType(input_type.get_element_type(), W_shape), 0.0, std::sqrt(1.0/input_size));
    auto b = device.backend().zeros(tensor::DataType(input_type.get_element_type(), b_shape));
    return {std::move(W), std::move(b)};
  };
}

namespace activation_func {
  static inline ActivationFunc relu = [](compiler::Device& device, compiler::Node affine) -> compiler::Node {
    return device.max(affine, 0);
  };
  
  static inline ActivationFunc softmax = [](compiler::Device& device, compiler::Node affine) -> compiler::Node {
    return device.softmax(affine);
  };
}

namespace loss_func {
  static inline LossFunc cross_entropy = [](compiler::Device& device, compiler::Node predicted, compiler::Node label) -> compiler::Node {
    return device.mean(-device.sum(label * predicted.log(), {1}));
  };
}

static inline auto dense = network::layer_def("Dense", [](compiler::Device& device, LayerPtr self, LayerPtr x, int output_size, DenseInitializer& initializer) {
  self->set_meta("output_size", std::to_string(output_size));
  auto X = x->out();
  auto [W_init, b_init] = initializer(device, X.get_datatype(), output_size);
  auto W = self->add_param(std::move(W_init));
  auto b = self->add_param(std::move(b_init));
  return device.matmul(X,W) + b;
});

static inline auto activation = network::layer_def("Activation", [](compiler::Device& device, LayerPtr self, LayerPtr x, ActivationFunc& func) {
  return func(device, x->out());
});

using ParamsVector = std::vector<compiler::Var>;

class Model;

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void optimize(compiler::Device& device, Model& model, ParamsVector& params_vec) = 0;
};

using OptimizerPtr = std::unique_ptr<Optimizer>;

namespace optimizer {
  class SGD : public Optimizer {
  public:
    SGD(double learning_rate) :
      learning_rate(learning_rate) {}

    void optimize(compiler::Device& device, Model& model, ParamsVector& params_vec) override {
      for (auto& params : params_vec) {
        if (!params.is_nograd())
          params.get_tensor() -= learning_rate * params.get_grad();
      }
    }

    double get_learning_rate() const {
      return learning_rate;
    }
  private:
    double learning_rate;
  };

  static inline OptimizerPtr sgd(double learning_rate) {
    return std::make_unique<SGD>(learning_rate);
  }
}

class Model {
public:
  using InputDefsMap = std::map<std::string,  compiler::PlaceHolder>;
  using InputsMap = std::map<std::string, tensor::Tensor>;
  Model(compiler::Device& device, const InputDefsMap& input_defs, const ParamsVector& params_vec, LayerPtr output, LossFunc loss, OptimizerPtr&& optimizer) :
    device(device), input_defs(input_defs), params_vec(params_vec), output(output), 
      loss(loss), optimizer(std::move(optimizer)), 
      label_placeholder(device.placeholder(output->out().get_datatype())) {
    setup_loss();
  }

  tensor::Tensor run(InputsMap&& inputs) {
    verify_inputs(inputs);
    for (auto& [name, tensor] : inputs) {
      input_defs.at(name).set_tensor(std::move(tensor));
    }
    auto program = device.compile(output->out());
    return program->forward();
  }

  double optimize(InputsMap&& inputs, tensor::Tensor&& label) {
    verify_inputs(inputs);
    label_placeholder.set_tensor(std::move(label));
    for (auto& [name, tensor] : inputs) {
      input_defs.at(name).set_tensor(std::move(tensor));
    }
    auto program = device.compile(loss_value);
    auto cur_loss = program->forward();
    program->backward();
    optimizer->optimize(device, *this, params_vec);
    return cur_loss(0).cast<double>();
  }

  void set_loss(LossFunc&& func) {
    this->loss = std::move(func);
    setup_loss();
  }

  void set_optimizer(OptimizerPtr&& optimizer) {
    this->optimizer = std::move(optimizer);
  }

  const ParamsVector& get_params_vec() const {
    return params_vec;
  }

  LayerPtr get_output() const {
    return output;
  }
private:
  void verify_inputs(const InputsMap& inputs) {
     assert(std::all_of(inputs.begin(), inputs.end(), [&](auto& it) -> bool {
      return input_defs.count(it.first);
    }) && "invalid input given");
    assert(inputs.size() == input_defs.size() && "not all input given");
  }

  void setup_loss() {
    if (loss) {
      loss_value = loss(device, output->out(), label_placeholder);
    }
  }

  compiler::Device& device;
  InputDefsMap input_defs;
  ParamsVector params_vec;
  LayerPtr output;
  LossFunc loss;
  compiler::Node loss_value;
  compiler::PlaceHolder label_placeholder;
  OptimizerPtr optimizer;
};

using ModelPtr = std::shared_ptr<Model>;

static inline ModelPtr finalize(LayerPtr final, LossFunc loss = nullptr, OptimizerPtr&& optimizer = nullptr) {
  assert(final->get_num_outputs() == 1 && "number of outputs of final layer must be one");
  std::set<Layer*> visited;
  std::map<std::string, compiler::PlaceHolder> inputs;
  std::vector<compiler::Var> params_vec;
  const auto dfs = [&](auto&& self, Layer& cur) -> void {
    if (visited.count(&cur)) { return; }
    if (cur.get_type() == LayerType::Input) {
      assert(!inputs.count(cur.get_input_name()) && "duplicate input name");
      inputs.emplace(cur.get_input_name(), cur.get_input_placeholder());
    }
    for (auto var : cur.get_params()) {
      params_vec.push_back(var);
    }
    for (auto& child : cur.ins()) {
      self(self, *child);
    }
  };
  dfs(dfs, *final);
  return std::make_shared<Model>(final->get_device(), inputs, params_vec, final, loss, std::move(optimizer));
}

}
}
}
