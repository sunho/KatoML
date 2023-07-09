#pragma once
#include "catch2/catch_approx.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mlsupport/mlsupport.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <katoml/mltensor/mltensor.hpp>

struct EqualsTensorMatcher : Catch::Matchers::MatcherGenericBase {
  EqualsTensorMatcher(const katoml::tensor::Tensor& tensor):
    tensor(tensor.copy()) {}

  bool match(const katoml::tensor::Tensor& other) const {
    if (tensor.get_shape() != other.get_shape())
      return false;
    bool ok = true;
    tensor.iterate_slow([&](const auto& index) -> void {
      ok &= tensor.at_slow(index) == other.at_slow(index);
    });
    return ok;
  }

  std::string describe() const override {
    return "Equals: " + katoml::to_string(tensor);
  }

private:
   katoml::tensor::Tensor tensor; 
};

static inline auto EqualsTensor(const katoml::tensor::Tensor& tensor) -> EqualsTensorMatcher {
  return EqualsTensorMatcher(tensor);
}

struct ApproxEqualsTensorMatcher : Catch::Matchers::MatcherGenericBase {
  ApproxEqualsTensorMatcher(const katoml::tensor::Tensor& tensor):
    tensor(tensor.copy()) {}

  bool match(const katoml::tensor::Tensor& other) const {
    if (tensor.get_shape() != other.get_shape())
      return false;
    bool ok = true;
    tensor.iterate_slow([&](const auto& index) -> void {
      ok &= tensor.at_slow(index).template cast<double>() == Catch::Approx(other.at_slow(index).template cast<double>());
    });
    return ok;
  }

  std::string describe() const override {
    return "Equals: " + katoml::to_string(tensor);
  }

private:
   katoml::tensor::Tensor tensor; 
};


static inline auto ApproxTensor(const katoml::tensor::Tensor& tensor) -> ApproxEqualsTensorMatcher {
  return ApproxEqualsTensorMatcher(tensor);
}
