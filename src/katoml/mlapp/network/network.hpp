#pragma once

#include "katoml/mlcompiler/device.hpp"
#include <functional>
#include <memory>

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

thread_local Context __thread_context;

static inline void set_thread_context(Context&& context) {
  __thread_context = std::move(context);
}

class Layer;
using LayerPtr = std::shared_ptr<Layer>;

class Layer {
public:
  Layer(const std::string& name, std::vector<network::LayerPtr> inputs) : 
    name(name), inputs(inputs) {}

  size_t get_num_inputs() const {
    return inputs.size();
  }

  const std::string& get_name() const {
    return name;
  }

  const std::vector<compiler::Node>& outs() const {
    return outputs;
  }

  void set_outs(const std::vector<compiler::Node>& outputs_) {
    this->outputs = outputs_;
  }

  compiler::Node out() const {
    return outs()[0];
  }
private:
  std::string name;
  std::vector<network::LayerPtr> inputs;
  std::vector<compiler::Node> outputs;
};

// template<typename ...Vargs>
// using LayerDefFunc = std::function<std::vector<compiler::Node>(compiler::Device& device, Vargs...)>;

// template<typename ...Vargs>
// using LayerDefSingleRetFunc = std::function<compiler::Node(compiler::Device& device, Vargs...)>;

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
    auto self = std::make_shared<Layer>(name, inputs);
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

static inline LayerDef input = network::layer_def("Input", [](compiler::Device& device, LayerPtr self, tensor::DataType datatype) -> compiler::Node {
  return device.placeholder(datatype);
});

static inline auto dense = network::layer_def("Dense", [](compiler::Device& device, LayerPtr self, LayerPtr x, size_t size) -> compiler::Node {
  return device.zeros_f32(1);
});

}
}
}
