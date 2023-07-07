#include "katoml/mlcompiler/mlcompiler.hpp"
#include "katoml/mltensor/mltensor.hpp"
#include "katoml/mlapp/mlapp.hpp"
#include "katoml/mltensor/types.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;
using namespace katoml::app;

static auto device = construct_device();

TEST_CASE( "Basic use case", "[cpu]" ) {
  auto custom = [&](Device& device, network::LayerPtr self, network::LayerPtr in1, network::LayerPtr in2) {
    // auto param = self->create_param("name", ElementType::Float32, Shape({1,2}), initializer);
    auto node = in1->outs()[0];
    return device.softmax(node);
  };

  auto custom_layer = network::layer_def("Custom", custom);

  network::Context ctx(*device);
  // implicit context API
  network::set_thread_context(std::move(ctx)); // or set_global_conext(ctx); default to global context if no thread context exists
  auto x = network::input(DataType(ElementType::Float32, Shape({1})));
  x = custom_layer(x, x);
  x = network::dense(x, 300);
  x = network::dense(x, 10);
  // auto model = network::finialize(x);
  
  // // explicit context API
  // auto x = network::input(ctx, shape);
  // x = custom_layer(ctx, x, x);
  // x = network::dense(ctx, x, 300, initializer, relu);
  // x = network::dense(ctx, x, 10, initializer, softmax);
  // auto model = network::finialize(ctx, x);
}

