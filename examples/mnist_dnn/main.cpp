#include "katoml/mltensor/types.hpp"
#include <katoml/mlcompiler/mlcompiler.hpp>
#include <katoml/mlapp/mlapp.hpp>
#include <katoml/mltensor/mltensor.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;
using namespace katoml::app;

const int default_batch_size = 128;
const float learing_rate = 0.1;

auto device = construct_device();
auto& backend = device->backend();

std::tuple<Tensor, Tensor, std::vector<int>> pick_data(MNistLoader& loader, size_t batch_size) {
  std::uniform_int_distribution<int> gen(0, loader.size()-1); 
  std::vector<std::vector<float>> X;
  std::vector<std::vector<float>> y;
  X.reserve(batch_size), y.reserve(batch_size);
  std::vector<int> labels;
  for (int i=0;i<batch_size;i++){
    int id = gen(rng);
    X.push_back(loader.images(id));
    std::vector<float> ans(10);
    ans[loader.labels(id)] = 1.0;
    y.push_back(ans);
    labels.push_back(loader.labels(id));
  }
  return {backend.from_vector(X), backend.from_vector(y), labels};
}

Var xavier_norm_var(Shape shape) {
  auto rand = backend.uniform(DataType(ElementType::Float32, shape), -1.0, 1.0);
  return device->var(rand / std::sqrt(shape[0]));
}

struct MNistDigitNetwork {
  MNistDigitNetwork() : 
    X(device->placeholder_f32(Shape::Any, 784)),
    y(device->placeholder_f32(Shape::Any, 10)),
    W(xavier_norm_var(Shape({784,784}))), 
    W2(xavier_norm_var(Shape({784,10}))), 
    b(xavier_norm_var(Shape({784}))),
    b2(xavier_norm_var(Shape({10}))) {}

  Node forward() {
    auto hidden = device->max(device->matmul(X,W) + b, device->zeros_f32(1));
    return device->softmax(device->matmul(hidden, W2) + b2);
  }

  Node cross_entropy() {
    auto y_ = forward();
    return device->mean(-device->sum(y * y_.log(), {1}));
  }

  std::vector<int> predict(const Tensor& images) {
    std::vector<int> res;
    size_t batch_size = images.get_shape()[0];
    X.set_tensor(images.copy());
    auto predicted = device->compile(forward())->forward();
    for (int i=0;i<batch_size;i++){
      std::pair<float, int> maxi = {-1.0, 0};
      for (int j=0;j<10;j++){
        maxi = std::max(maxi, {predicted.at_typed<float>(i,j),j});
      }
      res.push_back(maxi.second);
    }
    return res;
  }

  void train(const Tensor& images, const Tensor& label, size_t batch_size) {
    X.set_tensor(images.copy());
    y.set_tensor(label.copy());
    auto program = device->compile(cross_entropy());
    std::cout << "loss:" << program->forward() << "\n";
    program->backward();
    W.set_tensor(W.get_tensor() - W.get_grad()*learing_rate);
    b.set_tensor(b.get_tensor() - b.get_grad()*learing_rate);
    W2.set_tensor(W2.get_tensor() - W2.get_grad()*learing_rate);
    b2.set_tensor(b2.get_tensor() - b2.get_grad()*learing_rate);
  }

  PlaceHolder X, y;
  Var W, W2, b, b2;
};

int main() {
  MNistLoader data_loader("train.images", "train.label");

  MNistDigitNetwork model;
  for (int i=0;i<10000;i++){
    auto [images, labels, _] = pick_data(data_loader, default_batch_size);
    model.train(images, labels, default_batch_size);
    auto [images2, __, labels2] = pick_data(data_loader, default_batch_size);
    auto predicted = model.predict(images2);
    int cnt = 0;
    for (int i=0;i<predicted.size();i++) if (predicted[i] == labels2[i])  cnt ++;
    std::cout << "accuracy: " << (double) cnt / predicted.size() << "\n";
  }   
}