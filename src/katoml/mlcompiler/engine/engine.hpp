#pragma once

#include "../ir/ir.hpp"
#include <type_traits>

namespace katoml {
namespace compiler {
namespace engine {

class Engine {
public:
  Engine(tensor::Backend& backend) 
    : backend(backend) {}
  virtual ~Engine() = default;

  class Program {
  public:
    virtual ~Program() = default;
    virtual Tensor forward() = 0;
    virtual bool backward() = 0;
    virtual void reset() = 0;
  };

  virtual std::unique_ptr<Program> compile(ir::Value output) = 0;
  Tensor evaluate(ir::Value value) {
    return compile(value)->forward();
  }
protected:
  tensor::Backend& backend;
};

}
}
}