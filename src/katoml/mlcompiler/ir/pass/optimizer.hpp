#pragma once

#include <memory>
#include <unordered_map>
#include <set>
#include <list>
#include "../value.hpp"
#include "katoml/mlcompiler/ir/builder.hpp"
#include "katoml/mlcompiler/ir/types.hpp"

namespace katoml {
namespace compiler {
namespace ir {

class Pass {
public:
  virtual ~Pass() {}
  virtual Value run(Value value) = 0;
};

class NodeCombinePass : public Pass {
public:
  Value run(Value value) {
    if (!value.is_node())
      return value;
    auto& node = *value.as_node();
    visit(node);
    auto it = replaced.find(node.get_id());
    if (it != replaced.end()) {
      return it->second;
    }
    return value;
  }
private:
  void visit(Node& node) {
    if (visited.count(node.get_id())) return;
    visited.insert(node.get_id());
    for (int i=0;i<node.get_num_args();i++){
      auto& arg = node.get_args()[i];
      if (arg.is_node()) {
        visit(*arg.as_node());
      }
    }
  
    combine(node);

    for (int i=0;i<node.get_num_args();i++){
      auto& arg = node.get_args()[i];
      if (arg.is_node()) {
        auto it = replaced.find(arg.as_node()->get_id());
        if (it != replaced.end()) 
          arg = it->second;
      }
    }
  }

  void combine(Node& node) {
    switch (node.get_opcode()) {
    case Opcode::Log: {
      combine_log(node);
      break;
    }
    default: {
      break;
    }
    }
  }

  void combine_log(Node& node) {
    auto val = node.get_args()[0];
    if (val.is_node() && val.as_node()->get_opcode() == Opcode::SoftMax) {
      replace(node, Builder::LogSoftMax(val.as_node()->get_args()[0]));
    }
  }

  void replace(Node& node, Value value) {
    replaced[node.get_id()] = value;
  }

  std::set<uint64_t> visited;
  std::unordered_map<uint64_t, Value> replaced;
};

class PassManager {
public:
  PassManager() {}

  void add_pass(std::unique_ptr<Pass>&& pass) {
    passes.emplace_back(std::move(pass));
  }

  Value optimize(Value value) {
    for (auto& pass : passes) {
      value = pass->run(value);
    }
    return value;
  }
private:
  std::list<std::unique_ptr<Pass>> passes;
};

static inline std::unique_ptr<PassManager> construct_default_pass_manager() {
  auto manager = std::make_unique<PassManager>();
  manager->add_pass(std::make_unique<NodeCombinePass>());
  return manager;
}

}
}
}