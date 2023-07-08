#include "katoml/mlapp/network/network.hpp"
#include "katoml/mlcompiler/mlcompiler.hpp"
#include "katoml/mltensor/mltensor.hpp"
#include "katoml/mlapp/mlapp.hpp"
#include "katoml/mltensor/types.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace katoml;
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
    model->run(network::IM().set("input", device->backend().zeros_f32(1,10)).move());
  }

  {
    model->optimize(network::IM().set("input", device->backend().zeros_f32(1,10)).move(), device->backend().zeros_f32(10));
  }
}

TEST_CASE("[mlapp] Simple linear model") {
  network::Context ctx(*device);
  network::set_thread_context(std::move(ctx));

  auto x = network::input("input", DataType(ElementType::Float64, Shape({Shape::Any, 3})));
  x = network::dense(x, 1, network::initializer::constant(device->backend().tensor<double>({{0.1},{0.2},{0.3}}), device->backend().tensor<double>({0.8})));
  auto model = network::finalize(x, network::loss_func::mse, network::optimizer::sgd(1e-2));
  REQUIRE(to_string(model->get_output()->out()) == "Add(MatMul(Var(null), Var([[0.1], [0.2], [0.3]])), Var([0.8]))");

  auto data = device->backend().tensor<double>({
    {1.0,2.0,3.0},
    {4.0,5.0,6.0},
    {6.0,7.0,8.0}
  });
  auto label = device->backend().tensor<double>({{7.71},{11.61},{14.21}});
  std::vector<double> losses;
  for (int i=0;i<4;i++) {
    auto inputs = network::IM(); 
    inputs.set("input", data.copy());
    losses.push_back(model->optimize(std::move(inputs), label.copy()));
  }
  // FIXME: tensorflow is slightly (by 1e-5) better like a constant bias. Could this be just floating point error?
  std::vector<double> ans = {56.484100000000005, 22.527244170948226, 9.840964244481372, 5.084582271033033};
  for (int i=0;i<4;i++){
    REQUIRE(losses[i] == Catch::Approx(ans[i]));
  }
}

