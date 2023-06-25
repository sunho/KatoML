#pragma once
#include "types.hpp"
#include "value.hpp"

namespace katoml {
namespace compiler {
namespace ir {

constexpr int MAX_NODE_ARGUMENTS = 5;

template<class Backend>
class Node {
public:
  Node() = default;
  Node(Opcode opcode, DataType datatype, const std::initializer_list<Value<Backend>>& values) : 
    id(next_id++), opcode(opcode), datatype(datatype), num_args(values.size()) {
    assert(values.size() <= MAX_NODE_ARGUMENTS);
    std::copy(values.begin(), values.end(), args.begin());
  }
  uint64_t get_id() const { return id; }
  Value<Backend>* get_args() { return args.data(); }
  const Value<Backend>* get_args() const { return args.data(); }
  size_t get_num_args() const { return num_args; }
  Opcode get_opcode() const { return opcode; }
  DataType get_datatype() const { return datatype; }
private:
  uint64_t id;
  Opcode opcode;
  DataType datatype;
  std::array<Value<Backend>, MAX_NODE_ARGUMENTS> args;
  size_t num_args;
  static std::atomic<uint64_t> next_id;
};

template<class Backend>
std::atomic<uint64_t> Node<Backend>::next_id = 1;

}
}
}