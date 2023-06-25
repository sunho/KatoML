#pragma once

#include "../ir/ir.hpp"
#include <type_traits>

namespace katoml {
namespace compiler {
namespace engine {

template <typename T, typename Backend>
concept IEngineProgram = requires(T v, ir::Value<Backend> val) {
  {v.reset()};
  {v.forward()} -> std::same_as<typename Backend::Tensor>;
  {v.backward()} -> std::same_as<bool>;
};

template <typename T, typename Backend>
concept IEngineImpl = requires(T v, ir::Value<Backend> val, typename T::Program program) {
  requires IEngineProgram<typename T::Program, Backend>;
  {v.compile(val)} -> std::same_as<typename T::Program>;
};

template<typename Backend, IEngineImpl<Backend> EngineImpl>
class Engine {
public:
  Engine(Backend& backend) : impl(backend) {}

  using Program = typename EngineImpl::Program;
  using ComputeGraph = typename EngineImpl::ComputeGraph;
  Program compile(ir::Value<Backend> output) {
    return impl.compile(output);
  }
  TTensor evaluate(ir::Value<Backend> value) {
    return compile(value).forward();
  }
private:
  EngineImpl impl;
};

}
}
}