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

class CPUExecutor final : public Executor {
public:
  CPUExecutor() = default;
  CPUExecutor(const CPUExecutor&) = delete;
  CPUExecutor& operator=(const CPUExecutor&) = delete;
  ~CPUExecutor() = default;
  CPUExecutor(CPUExecutor&& other) :
    tensors(std::move(other.tensors)), next_id(other.next_id.load()) {}
  void relase(ExecutorHandle handle) override {
    ASSERT(tensors.find(handle.get_id()) != tensors.end(), "double releasing handle")
    tensors.erase(handle.get_id());
  }
  ExecutorHandle allocate(const TensorDescriptor& desc) override {
    return ExecutorHandle(tensors.emplace(next_id++, desc).first->first);
  }
  void* get_data(ExecutorHandle handle) override {
    return tensors.at(handle.get_id()).get_data();
  }
  bool is_alive(ExecutorHandle handle) override {
    return tensors.find(handle.get_id()) != tensors.end();
  }
  bool binop(BinOpcode opcode, HandleView res, HandleView lhs, HandleView rhs) override {
    return call_with_type<bool>([&]<typename T>(type_wrapper<T>) {
      auto work = [&](auto operation) {
        IterUtils::per_element_bin_op<T, operation>(to_wview(res), to_rview(lhs), to_rview(rhs));
        return true;
      };
      switch(opcode) {
      case BinOpcode::add:
        return work([](T a, T b) { return a + b; });
      case BinOpcode::sub:
        return work([](T a, T b) { return a - b; });
      case BinOpcode::mul:
        return work([](T a, T b) { return a * b; });
      case BinOpcode::div: 
        return work([](T a, T b) { return a / b; });
      case BinOpcode::max:
        return work([](T a, T b) { return std::max(a, b); });
      case BinOpcode::min:
        return work([](T a, T b) { return std::min(a, b); });
      case BinOpcode::less:
        return work([](T a, T b) { return static_cast<T>(a < b); });
      case BinOpcode::less_eq:
        return work([](T a, T b) { return static_cast<T>(a <= b); });
      case BinOpcode::more:
        return work([](T a, T b) { return static_cast<T>(a > b); });
      case BinOpcode::more_eq:
        return work([](T a, T b) { return static_cast<T>(a >= b); });
      case BinOpcode::equal:
        return work([](T a, T b) { return static_cast<T>(a == b); });
      case BinOpcode::matmul:
        return matmul(res, lhs, rhs);
      }
    }, get_element_type(res));
  }
  bool uniop(UniOpcode opcode, HandleView res, HandleView val) override {
     return call_with_type<bool>([&]<typename T>(type_wrapper<T>) {
      auto work = [&](auto operation) {
        IterUtils::per_element_uni_op<T, operation>(to_wview(res), to_rview(val));
        return true;
      };
      switch(opcode) {
      case UniOpcode::copy:
        return work([](T a) { return a; });
      case UniOpcode::exp:
        return work([](T a) { return std::exp(a); });
      case UniOpcode::log:
        return work([](T a) { return std::log(a); });
      case UniOpcode::neg:
        return work([](T a) { return -a; });
      case UniOpcode::diag:
        return diag(res, val);
      }
    }, get_element_type(res));
  }
  bool reduceop(ReduceOpcode opcode, HandleView res, HandleView res_std, HandleView val, const AxisArray& axis) override {
    switch (opcode) {
      case ReduceOpcode::max:
        return reduce_max(res, res_std, val);
      case ReduceOpcode::min:
        return reduce_min(res, res_std, val);
      case ReduceOpcode::sum:
        return sum(res, res_std, val);
      case ReduceOpcode::mean:
        return mean(res, res_std, val, axis);
    }
    return true;
  }
  bool selfop(SelfOpcode opcode, HandleView res, HandleView rhs) override {
    return call_with_type<bool>([&]<typename T>(type_wrapper<T>) {
      auto work = [&](auto operation) {
        IterUtils::per_element_self_bin_op<T, operation>(to_wview(res), to_rview(rhs));
        return true;
      };
      switch(opcode) {
      case SelfOpcode::add_assign:
        return work([](T a, T b) { return a + b; });
      case SelfOpcode::sub_assign:
        return work([](T a, T b) { return a - b; });
      }
    }, get_element_type(res));
  }
  bool near_equals(bool& res, HandleView lhs, HandleView rhs) override {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return std::abs(a-b) < NEAR_EQUAL_EPS.cast<T>(); };
      res = IterUtils::all<T, operation>(to_rview(lhs), to_rview(rhs));
    }, get_element_type(lhs));
    return true;
  }
  bool clip(HandleView res, HandleView val, Constant mn, Constant mx) override {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, std::pair<T,T> lims) { return std::clamp(a, lims.first, lims.second); };
      IterUtils::per_element_uni_op<T, std::pair<T,T>, operation>(to_wview(res), to_rview(val), {mn.cast<T>(), mx.cast<T>()});
    }, get_element_type(res));
    return true;
  }
  bool fill(HandleView res, Constant val) override {
     call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T val) { return val; };
      IterUtils::per_element_self<T, T, operation>(to_wview(res), val.cast<T>());
    }, get_element_type(res));
    return true;
  }
  bool matmul(HandleView res, HandleView lhs, HandleView rhs) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      IterUtils::matmul<T>(to_wview(res), to_rview(lhs), to_rview(rhs));
    }, get_element_type(res));
    return true;
  }
  
  bool diag(HandleView res, HandleView val) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a) { return a; };
      IterUtils::per_diag_element_uni_op<T, operation>(to_wview(res), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool sum(HandleView res, HandleView res_std, HandleView val) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a+b; };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
    }, get_element_type(res));
    return true;
  }
  bool mean(HandleView res, HandleView res_std,HandleView val, const AxisArray& axis) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto operation = [](T a, T b) { return a+b; };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
      const auto div = [](T a, size_t cnt) { return a/cnt; };
      IterUtils::per_element_self<T, size_t, div>(to_wview(res), calculate_reduced_count(val.shape, axis));
    }, get_element_type(res));
    return true;
  }
  bool reduce_max(HandleView res, HandleView res_std, HandleView val) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto init = [](T) { return std::numeric_limits<T>::min(); };
      IterUtils::per_element_self<T, init>(to_wview(res));
      const auto operation = [](T a, T b) { return std::max(a,b); };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val)); 
    }, get_element_type(res));
    return true;
  }
  bool reduce_min(HandleView res, HandleView res_std, HandleView val) {
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
      const auto init = [](T) { return std::numeric_limits<T>::max(); };
      IterUtils::per_element_self<T, init>(to_wview(res));
      const auto operation = [](T a, T b) { return std::min(a,b); };
      IterUtils::reduce<T, operation>(to_wview(res_std), to_rview(val));
    }, get_element_type(res));
    return true;
  }
private:
  IterUtils::ROffsetView to_rview(HandleView view) {
    const auto& tensor = tensors.at(view.handle.get_id());
    return IterUtils::ROffsetView(tensor.get_data(), view.offset, view.shape, view.strides);
  }

  IterUtils::WOffsetView to_wview(HandleView view) {
    auto& tensor = tensors.at(view.handle.get_id());
    return IterUtils::WOffsetView(tensor.get_data(), view.offset, view.shape, view.strides);
  }
  ElementType get_element_type(HandleView view) {
    return tensors.at(view.handle.get_id()).get_descriptor().get_element_type();
  }

  std::map<size_t, CPUTensor> tensors;
  std::atomic<size_t> next_id{1};
};

static inline std::unique_ptr<Backend> construct_cpu_backend() {
  return std::make_unique<Backend>(std::make_unique<CPUExecutor>());
}

}
}