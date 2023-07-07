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

class Device : public GraphDevice {
public:
  Device(std::unique_ptr<tensor::Backend>&& backend, std::unique_ptr<engine::Engine>&& engine) : GraphDevice(std::move(backend)),
    pass_manager(std::move(ir::construct_default_pass_manager())), 
   engine(std::move(engine)) {}
  
  std::unique_ptr<engine::Engine::Program> compile(Node output) { 
    ir::Value optimized = pass_manager->optimize(output.get_value());
    // std::cout << pretty_indent(to_string(optimized)) << "\n";
    return engine->compile(optimized); 
  }

private:
  std::unique_ptr<engine::Engine> engine;
  std::unique_ptr<ir::PassManager> pass_manager;
};

static inline std::unique_ptr<Device> construct_device() {
  auto backend = tensor::construct_cpu_backend();
  auto engine = std::make_unique<engine::InterpEngine>(*backend);
  return std::make_unique<Device>(std::move(backend), std::move(engine));
}

}
}