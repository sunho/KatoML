#pragma once

#include "katoml/mlcompiler/ir/pass/optimizer.hpp"
#include "katoml/mlcompiler/utils/string_utils.hpp"
#include "katoml/mltensor/types.hpp"
#include "tensor.hpp"
#include "graph/graph.hpp"
#include "engine/engine.hpp"
#include "engine/interp/interp.hpp"
#include <katoml/mltensor/cpu_backend.hpp>

namespace katoml {
namespace compiler {

template<typename Engine, class Executor>
class IDevice : public tensor::Backend<Executor> {
public:
  using Backend = tensor::Backend<Executor>;
  using Tensor = typename Backend::Tensor;
  using Node = Node<IDevice, Backend>;
  using Var = Var<IDevice, Backend>;
  using PlaceHolder = PlaceHolder<IDevice, Backend>;
  using Program = typename Engine::Program;
  using ComputeGraph = typename Engine::ComputeGraph;

  IDevice(Executor&& executor) : Backend(std::move(executor)), 
    pass_manager(std::move(ir::construct_default_pass_manager<Backend>())), 
    graph_builder(*this), engine(*this) {}
  
  Program compile(Node output) { 
    ir::Value optimized = pass_manager->optimize(output.get_value());
    // std::cout << pretty_indent(to_string(optimized)) << "\n";
    return engine.compile(optimized); 
  }

  template<typename T>
  using TypedTensor = typename Backend::template TypedTensor<T>;

  GraphBuilder<IDevice, Backend>& graph() {
    return graph_builder;
  }
private:
  Engine engine;
  std::unique_ptr<ir::PassManager<Backend>> pass_manager;
  GraphBuilder<IDevice, Backend> graph_builder;
};

using Device = IDevice<engine::InterpEngine<tensor::CPUBackend>, tensor::CPUExecutor>;

static inline std::unique_ptr<Device> construct_device() {
  return std::make_unique<Device>(std::move(tensor::CPUExecutor()));
}

}
}