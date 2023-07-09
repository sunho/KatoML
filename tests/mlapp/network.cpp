#include "katoml/mlapp/network/network.hpp"
#include "katoml/mlcompiler/mlcompiler.hpp"
#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/mltensor.hpp"
#include "katoml/mlapp/mlapp.hpp"
#include "katoml/mltensor/types.hpp"
#include "../common.hpp"
#include <filesystem>
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
  // FIXME: tensorflow is slightly (by 1e-5) better by like a constant bias. Could this be really just floating point error?
  std::vector<double> ans = {56.484100000000005, 22.527244170948226, 9.840964244481372, 5.084582271033033};
  for (int i=0;i<4;i++){
    REQUIRE(losses[i] == Catch::Approx(ans[i]));
  }
}

TEST_CASE("[mlapp] Simple softmax model") {
  network::Context ctx(*device);
  network::set_thread_context(std::move(ctx));

  auto x = network::input("input", DataType(ElementType::Float64, Shape({Shape::Any, 5})));
  x = network::dense(x, 3, network::initializer::constant(device->backend().zeros_f64(5,3).fill(0.01f), device->backend().zeros_f64(3).fill(0.01f)));
  x = network::activation(x, network::activation_func::softmax);
  auto model = network::finalize(x, network::loss_func::cross_entropy, network::optimizer::sgd(1.0));

  auto data = device->backend().tensor<double>({
    {-1.1258398, -1.1523602, -0.2505786, -0.4338788,  0.8487104},
    { 0.6920092, -0.3160128, -2.1152194,  0.3222749, -1.2633348},
    { 0.3499832,  0.3081339,  0.1198415,  1.2376579,  1.1167772},
    {-0.2472782, -1.3526537, -1.6959312,  0.5666506,  0.7935084},
    { 0.5988395, -1.5550951, -0.3413604,  1.8530061,  0.7501895},
    {-0.5854976, -0.1733968,  0.1834779,  1.3893661,  1.5863342},
    { 0.9462984, -0.8436767, -0.6135831,  0.0315927,  1.0553575},
    { 0.1778437, -0.2303355, -0.3917544,  0.5432947, -0.3951575},
    { 0.2055257, -0.4503298,  1.5209795,  3.4105027, -1.5311843},
    {-1.2341350,  1.8197253, -0.5515287, -1.3253260,  0.1885536}
  });
  auto label = device->backend().tensor<double>({
    {1., 0., 0.},
    {1., 0., 0.},
    {0., 0., 1.},
    {1., 0., 0.},
    {0., 0., 1.},
    {0., 0., 1.},
    {0., 0., 1.},
    {1., 0., 0.},
    {0., 0., 1.},
    {1., 0., 0.}
  });
  std::vector<double> losses;
  for (int i=0;i<10;i++) {
    auto inputs = network::IM(); 
    inputs.set("input", data.copy());
    losses.push_back(model->optimize(std::move(inputs), label.copy()));
  }
  std::vector<double> ans = {1.0986121892929077, 0.49101758003234863, 0.3514918088912964, 0.2814021706581116, 0.23764991760253906};
  for (int i=0;i<5;i++){
    REQUIRE(losses[i] == Catch::Approx(ans[i]));
  }
}

TEST_CASE("[mlapp] Save and load model is working") {
  network::Context ctx(*device);
  network::set_thread_context(std::move(ctx)); // or set_global_conext(ctx); default to global context if no thread context exists
  auto x = network::input("input", DataType(ElementType::Float32, Shape({1, 10})));
  x = network::dense(x, 300, network::initializer::xavier); 
  auto model = network::finalize(x);
  model->save_params(std::filesystem::temp_directory_path() / "__mlapp_save_load_test_model.kato");
  auto p1 = model->get_params_vec()[0].get_tensor().copy();
  auto p2 = model->get_params_vec()[1].get_tensor().copy();
  model->get_params_vec()[0].set_tensor(device->backend().zeros(p1.get_datatype()));
  model->load_params(std::filesystem::temp_directory_path() / "__mlapp_save_load_test_model.kato");
  REQUIRE_THAT(model->get_params_vec()[0].get_tensor(), EqualsTensor(p1));
  REQUIRE_THAT(model->get_params_vec()[1].get_tensor(), EqualsTensor(p2));
}