#pragma once
#include "katoml/mltensor/types.hpp"
#include <katoml/mlcompiler/mlcompiler.hpp>
#include <katoml/mlapp/mlapp.hpp>
#include <katoml/mltensor/mltensor.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;
using namespace katoml::app;

static inline std::tuple<Tensor, Tensor, std::vector<int>> pick_mnist_data(MNistLoader& loader, Backend& backend, size_t batch_size) {
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

static inline std::vector<int> predcited_to_label(const Tensor& predicted) {
  std::vector<int> res;
  for (int i=0;i<predicted.get_shape()[0];i++){
    std::pair<float, int> maxi = {-1.0, 0};
    for (int j=0;j<predicted.get_shape()[1];j++){
      maxi = std::max(maxi, {predicted.at_typed<float>(i,j),j});
    }
    res.push_back(maxi.second);
  }
  return res;
}

static inline double match_percent(const std::vector<int>& predicted, const std::vector<int>& label) {
  int cnt = 0;
  for (int i=0;i<predicted.size();i++) if (predicted[i] == label[i]) cnt ++;
  return (double) cnt / predicted.size();
}