#include "../common.hpp"

const int default_batch_size = 128;
const float learing_rate = 0.1;

auto device = construct_device();
auto& backend = device->backend();

network::ModelPtr build_model(double learning_rate) {
  using namespace network;
  auto x = input("input", DataType(ElementType::Float32, Shape({Shape::Any, 784})));
  x = dense(x, 784, initializer::xavier);
  x = activation(x, activation_func::relu);
  x = dense(x, 10, initializer::xavier);
  x = activation(x, activation_func::softmax);
  return finalize(x, loss_func::cross_entropy, optimizer::sgd(learning_rate));
}

int main() {
  network::Context ctx(*device);
  network::set_thread_context(std::move(ctx));
  MNistLoader data_loader("train.images", "train.label");
  auto model = build_model(learing_rate);

  for (int i=0;i<10000;i++){
    auto [images, labels, _] = pick_mnist_data(data_loader, backend, default_batch_size);
    double loss = model->optimize({{"input", std::move(images)}}, std::move(labels));
    auto [images2, __, labels2] = pick_mnist_data(data_loader, backend, default_batch_size);
    auto predicted = predcited_to_label(model->run({{"input", std::move(images2)}}));
    std::cout << "accuracy: " << match_percent(predicted, labels2) << "\n";
  }
}
