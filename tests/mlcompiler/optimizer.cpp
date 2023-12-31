#include "katoml/mlcompiler/ir/pass/optimizer.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mlsupport/mlsupport.hpp"
#include "katoml/mltensor/cpu_backend.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mlcompiler/mlcompiler.hpp>

using namespace katoml;
using namespace katoml::compiler;
using namespace katoml::tensor;

static auto device = construct_device();
static auto default_pass_manager = ir::construct_default_pass_manager();

TEST_CASE("[mlcompiler] Log softmax combine pass") {
  auto node = device->log(device->softmax(device->zeros_f32(1)));
  REQUIRE(to_string(node) == "Log(SoftMax([0]))");
  auto optimized = default_pass_manager->optimize(node.get_value());
  REQUIRE(to_string(optimized) == "LogSoftMax([0])");
}

TEST_CASE("[mlcompiler] Log softmax combine pass on cross entropy") {
  auto y_ = device->placeholder_f32(Shape::Any, 2);
  y_.set_tensor(device->backend().zeros_f32(1, 2));
  auto y = device->softmax(device->zeros_f32(1, 2));
  auto loss = device->mean(-device->sum(y_ * y.log(), {1}));
  REQUIRE(to_string(loss) == "ReduceMean(Neg(ReduceSum(Mul(Var([[0, 0]]), Log(SoftMax([[0, 0]]))), axis=[1])), axis=[0])");
  auto optimized = default_pass_manager->optimize(loss.get_value());
  REQUIRE(to_string(optimized) == "ReduceMean(Neg(ReduceSum(Mul(Var([[0, 0]]), LogSoftMax([[0, 0]])), axis=[1])), axis=[0])");
}