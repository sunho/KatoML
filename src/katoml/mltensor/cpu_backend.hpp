#pragma once
#include <map>
#include <numeric>
#include <vector>
#include "core.hpp"
#include "iter_utils.hpp"
#include "katoml/mltensor/types.hpp"

namespace katoml {
namespace tensor {
class CPUTensor final {
public:
  CPUTensor(TensorDescriptor descriptor) 
    : descriptor(descriptor), data((descriptor.get_data_size() + 15)/16) {}
  TensorDescriptor get_descriptor() const { return descriptor; }
  const uint8_t* get_data() const { return (const uint8_t*)data.data(); }
  uint8_t* get_data() { return (uint8_t*)data.data(); }
private:
  TensorDescriptor descriptor{};
  std::vector<__int128_t> data;
};

class CPUExecutor final {
public:
  CPUExecutor() = default;
  CPUExecutor(const CPUExecutor&) = delete;
  CPUExecutor& operator=(const CPUExecutor&) = delete;
  ~CPUExecutor() = default;
  CPUExecutor(CPUExecutor&& other) :
    tensors(std::move(other.tensors)), next_id(other.next_id.load()) {}
    
  using Handle = size_t;
  struct HandleView {
    Handle handle{};
    size_t offset{};
    Shape shape{};
    Strides strides{};
  };
  void relase(Handle handle) {
    ASSERT(tensors.find(handle) != tensors.end(), "double releasing handle")
    tensors.erase(handle);
  }
  bool is_alive(Handle handle) {
    return tensors.find(handle) != tensors.end();
  }
  Handle allocate(TensorDescriptor descriptor) {
    return tensors.emplace(next_id++, descriptor).first->first;
  }
  void* get_data(Handle handle) {
    return tensors.at(handle).get_data();
  }
  bool add(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a + b; };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool add_assign(HandleView res, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a + b; };
      IterUtils::per_element_self_bin_op<T, operation>(to_wview(res), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool sub(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a - b; };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool sub_assign(HandleView res, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a - b; };
      IterUtils::per_element_self_bin_op<T, operation>(to_wview(res), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool mul(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a * b; };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool div(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a / b; };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool max(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return std::max(a, b); };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool min(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return std::min(a, b); };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool less(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return static_cast<T>(a < b); };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool less_eq(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return static_cast<T>(a <= b); };
      IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool equals(bool& res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a == b; };
      res = IterUtils::all<T, operation>(to_rview(lhs), to_rview(rhs));
    }, get_element_type(lhs));
    return true;
  }
  bool near_equals(bool& res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return std::abs(a-b) < NEAR_EQUAL_EPS.cast<T>(); };
      res = IterUtils::all<T, operation>(to_rview(lhs), to_rview(rhs));
    }, get_element_type(lhs));
    return true;
  }
  bool matmul(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      IterUtils::matmul<T>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  bool copy(HandleView res, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a) { return a; };
      IterUtils::per_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool diag(HandleView res, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a) { return a; };
      IterUtils::per_diag_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool fill(HandleView res, Constant val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T val) { return val; };
      IterUtils::per_element_self<T, T, operation>(to_wview(res), val.cast<T>());
    }, get_element_type(res));
    return true;
  }
  bool clip(HandleView res, HandleView val, Constant mn, Constant mx) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, std::pair<T,T> lims) { return std::clamp(a, lims.first, lims.second); };
      IterUtils::per_element_uni_op<T, std::pair<T,T>, operation>(to_wview(res), to_rview(val), {mn.cast<T>(), mx.cast<T>()});
    }, get_element_type(res));
    return true;
  }
  bool log(HandleView res, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a) { return std::log(a); };
      IterUtils::per_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool exp(HandleView res, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a) { return std::exp(a); };
      IterUtils::per_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool neg(HandleView res, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
    const auto operation = [](T a) { return -a; };
    IterUtils::per_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool sum(HandleView res, HandleView res_std, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a+b; };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool mean(HandleView res, HandleView res_std,HandleView val, const AxisArray& axis) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a+b; };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
      const auto div = [](T a, size_t cnt) { return a/cnt; };
      IterUtils::per_element_self<T, size_t, div>(to_wview(res), calculate_reduced_count(val.shape, axis));
    }, get_element_type(res));
    return true;
  }
  bool reduce_max(HandleView res, HandleView res_std, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto init = [](T) { return std::numeric_limits<T>::min(); };
      IterUtils::per_element_self<T, init>(to_wview(res));
      const auto operation = [](T a, T b) { return std::max(a,b); };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val)); 
    }, get_element_type(res));
    return true;
  }
  bool reduce_min(HandleView res, HandleView res_std, HandleView val) {
    call_with_type([&]<typename T>(type_wrapper<T>) {
      const auto init = [](T) { return std::numeric_limits<T>::max(); };
      IterUtils::per_element_self<T, init>(to_wview(res));
      const auto operation = [](T a, T b) { return std::min(a,b); };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
    }, get_element_type(res));
    return true;
  }
private:
  IterUtils::ROffsetView to_rview(HandleView view) {
    const auto& tensor = tensors.at(view.handle);
    return IterUtils::ROffsetView(tensor.get_data(), view.offset, view.shape, view.strides);
  }

  IterUtils::WOffsetView to_wview(HandleView view) {
    auto& tensor = tensors.at(view.handle);
    return IterUtils::WOffsetView(tensor.get_data(), view.offset, view.shape, view.strides);
  }
  ElementType get_element_type(HandleView view) {
    return tensors.at(view.handle).get_descriptor().get_element_type();
  }

  std::map<Handle, CPUTensor> tensors;
  std::atomic<size_t> next_id{1};
};
using CPUBackend = Backend<CPUExecutor>;
static inline CPUBackend construct_cpu_backend() {
  return CPUBackend(CPUExecutor());
}

}
}