#include "mlcompiler.hpp"
#include "katoml/mlcompiler/device.hpp"

std::atomic<uint64_t> katoml::compiler::ir::Node::next_id = 1;
std::unique_ptr<katoml::compiler::Device> katoml::compiler::default_device
  = katoml::compiler::construct_device();
