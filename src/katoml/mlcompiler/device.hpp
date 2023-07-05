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

template<typename Engine, class Backend>
class IDevice : public GraphDevice<Backend> {
public:
  using Tensor = typename Backend::Tensor;
  using Node = Node<GraphDevice<Backend>, Backend>;
  using Var = Var<GraphDevice<Backend>, Backend>;
  using PlaceHolder = PlaceHolder<GraphDevice<Backend>, Backend>;
  using Program = typename Engine::Program;
  using ComputeGraph = typename Engine::ComputeGraph;

  IDevice(Backend&& backend) : GraphDevice<Backend>(std::move(backend)),
    pass_manager(std::move(ir::construct_default_pass_manager<Backend>())), 
   engine(this->backend_) {}
  
  Program compile(Node output) { 
    ir::Value optimized = pass_manager->optimize(output.get_value());
    // std::cout << pretty_indent(to_string(optimized)) << "\n";
    return engine.compile(optimized); 
  }

  template<typename T>
  using TypedTensor = typename Backend::template TypedTensor<T>;
private:
  Engine engine;
  std::unique_ptr<ir::PassManager<Backend>> pass_manager;
};

using Device = IDevice<engine::InterpEngine<tensor::CPUBackend>, tensor::CPUBackend>;

static inline std::unique_ptr<Device> construct_device() {
  return std::make_unique<Device>(tensor::construct_cpu_backend());
}

}
}