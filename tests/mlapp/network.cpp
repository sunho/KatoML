#include "katoml/mlapp/network/network.hpp"
#include "katoml/mlcompiler/mlcompiler.hpp"
#include "katoml/mltensor/mltensor.hpp"
#include "katoml/mlapp/mlapp.hpp"
#include "katoml/mltensor/types.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;
using namespace katoml::app;

static auto device = construct_device();

TEST_CASE("[mlapp] Basic use case") {
  auto custom = [&](Device& device, network::LayerPtr self, network::LayerPtr in1, network::LayerPtr in2) {
    auto node = in1->outs()[0];
    return device.softmax(node);
  };

  auto custom_layer = network::layer_def("Custom", custom);

  network::Context ctx(*device);
  // implicit context API
  network::set_thread_context(std::move(ctx)); // or set_global_conext(ctx); default to global context if no thread context exists
  auto x = network::input("input", DataType(ElementType::Float32, Shape({1, 10})));
  x = custom_layer(x, x);
  x = network::dense(x, 300, network::initializer::xavier);
  x = network::activation(x, network::activation_func::relu);
  x = network::dense(x, 10, network::initializer::xavier);
  x = network::activation(x, network::activation_func::softmax);
  auto model = network::finalize(x, network::loss_func::cross_entropy, network::optimizer::sgd(0.1));
  REQUIRE(
    to_string(model->get_output()->out()) == 
    "SoftMax(Add(MatMul(Max(Add(MatMul(SoftMax(Var(null)), Var([Float32[10, 300]]"
    ")), Var([Float32[300]])), [0]), Var([Float32[300, 10]])), Var([0, 0, 0, 0, 0, "
    "0, 0, 0, 0, 0])))"
  );
  
  // // explicit context API
  // auto x = network::input(ctx, shape);
  // x = custom_layer(ctx, x, x);
  // x = network::dense(ctx, x, 300, initializer, relu);
  // x = network::dense(ctx, x, 10, initializer, softmax);
  // auto model = network::finialize(ctx, x);

  {
    network::Model::InputsMap inputs;
    inputs.emplace("input", device->backend().zeros_f32(1,10));
    model->run(std::move(inputs));
  }

  {
    network::Model::InputsMap inputs;
    inputs.emplace("input", device->backend().zeros_f32(1,10));
    model->optimize(std::move(inputs), device->backend().zeros_f32(10));
  }
}

