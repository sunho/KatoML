

TEST_CASE( "Basic use case", "[cpu]" ) {
  auto custom = [&](Device& device, network::Layer self, network::Layer in1, network::Layer in2) {
    auto param = self->create_param("name", ElementType::Float32, Shape({1,2}), initializer);
    auto node = in1->outs()[0];
    return device.softmax(node) * param;
  };
  auto custom_layer = network::layer_def("Custom", custom);

  network::Context ctx(device);
  // implicit context API
  network::set_thread_context(ctx); // or set_global_conext(ctx); default to global context if no thread context exists
  auto x = network::input(shape);
  x = custom_layer(x, x);
  x = network::dense(x, 300, initializer, relu);
  x = network::dense(x, 10, initializer, softmax);
  auto model = network::finialize(x);
  
  // explicit context API
  auto x = network::input(ctx, shape);
  x = custom_layer(ctx, x, x);
  x = network::dense(ctx, x, 300, initializer, relu);
  x = network::dense(ctx, x, 10, initializer, softmax);
  auto model = network::finialize(ctx, x);
}

