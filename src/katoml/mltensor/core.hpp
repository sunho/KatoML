#pragma once
#include <functional>
#include <optional>
#include <random>
#include <iostream>

#include "core.hpp"
#include "errors.hpp"
#include "katoml/mltensor/core.hpp"
#include "types.hpp"
#include "iter_utils.hpp"

namespace katoml {
namespace tensor {

static int64_t MAX_TENSOR_LOG_LIMIT = 128;

template <typename T> concept signed_type = std::is_signed_v<T>; 

template <typename T>
concept IHandle = requires(T v) {
  {v.get_ndims()} -> std::convertible_to<size_t>;
  {v.get_element_type()} -> std::convertible_to<ElementType>;
  {v.get_shape()} -> std::convertible_to<Shape>;
  {v.get_strides()} -> std::convertible_to<Strides>;
};

template <typename T>
concept IBackend = requires(T v, typename T::Handle h) {
  requires IHandle<typename T::Handle>;
};

template <typename T>
class type_wrapper {
public:
};

static inline void call_with_type(auto&& func, ElementType element_type) {
  switch (element_type) {
  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  case ElementType::enum_name: \
  func(type_wrapper<tyty>());\
  break;
  #include "element_type.inc"
  #undef ELEMENT_TYPE
  case ElementType::None:
    assert(false);
    break;
  }
}

#define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
static inline ElementType find_element_type(type_wrapper<tyty>) {\
  return ElementType::enum_name;\
}
#include "element_type.inc"
#undef ELEMENT_TYPE

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

class Constant {
enum class ConstantType { None, Min, Max, Value };
public:
  Constant() = default;
  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) Constant(tyty val) : val(val), type(ConstantType::Value) {}
  #include "element_type.inc"
  #undef ELEMENT_TYPE
  template<typename T> T cast() const {
    if (type == ConstantType::None)
      throw NullConstantError();
    if (type == ConstantType::Max) 
      return std::numeric_limits<T>::max();
    if (type == ConstantType::Min) 
      return std::numeric_limits<T>::min();
    return std::visit(overloaded {
      [](std::monostate) { 
        panic("monostate tensor constant");
        return static_cast<T>(0);
      },
      [](auto v) {return static_cast<T>(v); }
    }, val);
  }
  static inline Constant max() {
    return Constant(ConstantType::Max);
  }
  static inline Constant min() {
    return Constant(ConstantType::Min);
  }
private:
  Constant(ConstantType type) : type(type) {}
  using Inner = std::variant<
    #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) tyty,
    #include "element_type.inc"
    #undef ELEMENT_TYPE
    std::monostate
  >;
  ConstantType type { ConstantType::None };
  Inner val;
};

static Constant NEAR_EQUAL_EPS = 1e-10; 

template<IBackend Backend>
class TensorBase {
using BackendHandle = typename Backend::Handle;
public:
  TensorBase(Backend& backend, BackendHandle handle, bool is_view = false) : backend(backend), handle(handle), is_view(is_view) {}
  TensorBase(const TensorBase& other) = delete;
  TensorBase& operator=(const TensorBase& other) = delete;
  TensorBase(TensorBase&& other) : backend(other.backend), handle(other.handle), is_view(other.is_view) {
    other.handle = {};
  }
  TensorBase& operator=(TensorBase&& other) {
    backend = std::move(other.backend);
    handle = other.handle;
    is_view = other.is_view;
    other.handle = {};
    return *this;
  }
  ~TensorBase() {
    if (handle && !is_view) {
      backend.get().release(handle);
    }
  }
  TensorBase copy() const { return TensorBase(backend, backend.get().copy(handle)); }
  int get_ndims() const { return handle.get_ndims(); }
  ElementType get_element_type() const { return handle.get_element_type(); }
  Shape get_shape() const { return handle.get_shape(); }
  Strides get_strides() const { return handle.get_strides(); }
  DataType get_datatype() const { return DataType(get_element_type(), get_shape()); }
  TensorDescriptor get_descriptor() const { 
    return TensorDescriptor(handle.get_element_type(), handle.get_shape(), handle.get_strides()); 
  }
  template<typename T, signed_type... Index>
  T at(Index... index) const {
    uint8_t* data = reinterpret_cast<uint8_t*>(get_data());
    int i = 0;
    size_t offset = 0;
    ((offset += index * handle.get_strides()[i++]), ...);
    ASSERT(i == get_ndims(), "tensor at index not matching dimension");
    return *reinterpret_cast<const T*>(data + offset);
  }
  template<typename T, signed_type... Index>
  T& at(Index... index) {
    uint8_t* data = reinterpret_cast<uint8_t*>(get_data());
    int i = 0;
    size_t offset = 0;
    ((offset += index * handle.get_strides()[i++]), ...);
    ASSERT(i == get_ndims(), "tensor at index not matching dimension");
    return *reinterpret_cast<T*>(data + offset);
  }
  TensorBase operator+(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().add(handle, rhs.handle));
  }
  TensorBase operator-(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().sub(handle, rhs.handle));
  }
  TensorBase operator*(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().mul(handle, rhs.handle));
  }
  TensorBase operator/(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().div(handle, rhs.handle));
  }
  TensorBase operator<(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().less(handle, rhs.handle));
  }
  TensorBase operator<=(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().less_eq(handle, rhs.handle));
  }
  TensorBase operator>(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().less(rhs.handle, handle));
  }
  TensorBase operator>=(const TensorBase& rhs) const {
    backend_check(rhs);
    return TensorBase(backend.get(), backend.get().less_eq(rhs.handle, handle));
  }
  TensorBase max(const TensorBase& rhs) const {
    return TensorBase(backend.get(), backend.get().max(handle, rhs.handle));
  }
  TensorBase min(const TensorBase& rhs) const {
    return TensorBase(backend.get(), backend.get().min(handle, rhs.handle));
  }
  TensorBase operator-() const {
    return TensorBase(backend.get(), backend.get().neg(handle));
  }
  TensorBase matmul(const TensorBase& other) const {
    backend_check(other);
    return TensorBase(backend.get(), backend.get().matmul(handle, other.handle));
  }
  TensorBase log() const {
    return TensorBase(backend.get(), backend.get().log(handle));
  }
  TensorBase exp() const {
    return TensorBase(backend.get(), backend.get().exp(handle));
  }
  TensorBase clip(Constant mn, Constant mx) const {
    return TensorBase(backend.get(), backend.get().clip(handle, mn, mx));
  }
  void fill(Constant val) {
    backend.get().fill(handle, val);
  }
  TensorBase diag() const {
    return TensorBase(backend.get(), backend.get().diag(handle));
  }
  TensorBase sum(const AxisArray& axis = AllAxis) const {
    return TensorBase(backend.get(), backend.get().sum(handle, axis));
  }
  TensorBase mean(const AxisArray& axis = AllAxis) const {
    return TensorBase(backend.get(), backend.get().mean(handle, axis));
  }
  TensorBase max(const AxisArray& axis = AllAxis) const {
    return TensorBase(backend.get(), backend.get().max(handle, axis));
  }
  TensorBase min(const AxisArray& axis = AllAxis) const {
    return TensorBase(backend.get(), backend.get().min(handle, axis));
  }
  void reshape(Shape shape) {
    handle = backend.get().reshape(handle, shape);
  }
  TensorBase reshaped(Shape shape) const {
    auto res = backend.get().reshape(handle, shape);
    if (handle == res) return TensorBase(backend.get(), res, true);
    return TensorBase(backend.get(), res);
  }
  void transpose() {
    handle = backend.get().transpose(handle);
  }
  TensorBase transposed() const {
    auto res = backend.get().transpose(handle);
    if (handle == res) return TensorBase(backend.get(), res, true);
    return TensorBase(backend.get(), res);
  }
  void extend_axis() {
    reshape(get_shape().extend_axis());
  }
  TensorBase axis_extended() const {
    return reshaped(get_shape().extend_axis());
  }
  TensorBase& operator+=(const TensorBase& rhs) {
    backend_check(rhs);
    BackendHandle tmp = handle;
    handle = backend.get().add(tmp, rhs.handle);
    backend.get().release(tmp);
    return *this;
  }
  TensorBase& operator-=(const TensorBase& rhs) {
    backend_check(rhs);
    BackendHandle tmp = handle;
    handle = backend.get().sub(tmp, rhs.handle);
    backend.get().release(tmp);
    return *this;
  }
  bool operator==(const TensorBase& other) const {
    return backend.get().equals(handle, other.handle);
  }
  bool near_equals(const TensorBase& other) const {
    return backend.get().near_equals(handle, other.handle);
  }
  friend std::ostream& operator<<(std::ostream& os, const TensorBase& tensor) {
    std::ostringstream ss("");
    call_with_type([&]<typename T>(type_wrapper<T>) {
      IterUtils::ROffsetView rview(tensor.get_data(),tensor.get_descriptor());
      const auto level_in = [&](int level) {
        ss << "[";
      };
      const auto level_out = [&](int level, bool last) {
        if (last)
          ss << "]";
        else
          ss << "], ";
      };
      const auto read = [&](T val, bool last) {
        if (last)
          ss << val;
        else
          ss << val << ", ";
      };
      IterUtils::per_element_read<T>(rview, level_in, level_out, read);
    }, tensor.get_element_type());
    if (ss.tellp() > MAX_TENSOR_LOG_LIMIT) {
      return os << "[" << tensor.get_datatype() << "]";
    }
    os << ss.str();
    return os;
  }
protected:
  void* get_data() {
    return backend.get().get_data(handle);
  }
  const void* get_data() const {
    return backend.get().get_data(handle);
  }

  void backend_check(const TensorBase& other) const {
    CHECK_OR_THROW(&backend.get() == &other.backend.get(), BackendMismatchError)
  }

  std::reference_wrapper<Backend> backend;
  bool is_view;
  BackendHandle handle;
};

template<IBackend Backend, typename T>
class TypedTensorBase : public TensorBase<Backend> {
using BackendHandle = typename Backend::Handle;
public:
  TypedTensorBase() = default;
  TypedTensorBase(TensorBase<Backend>&& other) : TensorBase<Backend>(std::move(other)) {
    CHECK_OR_THROW(find_element_type(type_wrapper<T>()) == this->get_element_type(), TensorTypeError)
  }
  TypedTensorBase(TypedTensorBase&& other) : TensorBase<Backend>(std::move(other)) {
    CHECK_OR_THROW(find_element_type(type_wrapper<T>()) == this->get_element_type(), TensorTypeError)
  }
  TypedTensorBase(const TypedTensorBase& other) = delete;
  TypedTensorBase& operator=(const TypedTensorBase& other) = delete;
  TypedTensorBase& operator=(TensorBase<Backend>&& other) {
    TensorBase<Backend>::operator=(std::move(other));
    return *this;
  }
  TypedTensorBase& operator=(TypedTensorBase&& other) {
    TensorBase<Backend>::operator=(std::move(other));
    return *this;
  }
  template<typename F, signed_type... Index> F at(Index... index) const = delete;
  template<typename F, signed_type... Index> F& at(Index... index) = delete;
  template<signed_type... Index>
  T operator()(Index... index) const { return TensorBase<Backend>::template at<T>(index...); }
  template<signed_type... Index>
  T& operator()(Index...index) { return TensorBase<Backend>::template at<T>(index...); }
  TypedTensorBase copy() const { 
    return std::move(TensorBase<Backend>::copy());
  }
  TensorBase<Backend> operator+(const TensorBase<Backend>& rhs) const = delete;
  TypedTensorBase operator+(const TypedTensorBase& rhs) const {
    return TensorBase<Backend>::operator+(rhs);
  }
  TensorBase<Backend> operator-(const TensorBase<Backend>& rhs) const = delete;
  TypedTensorBase operator-(const TypedTensorBase& rhs) const {
    return TensorBase<Backend>::operator-(rhs);
  }
  TensorBase<Backend> operator*(const TensorBase<Backend>& rhs) const = delete;
  TypedTensorBase operator*(const TypedTensorBase& rhs) const {
    return TensorBase<Backend>::operator*(rhs);
  }
  TensorBase<Backend> operator/(const TensorBase<Backend>& rhs) const = delete;
  TypedTensorBase operator/(const TypedTensorBase& rhs) const {
    return TensorBase<Backend>::operator/(rhs);
  }
  TypedTensorBase operator-() const {
    return TensorBase<Backend>::operator-();
  }
  TypedTensorBase matmul(const TypedTensorBase& other) const {
    return TensorBase<Backend>::matmul((const TensorBase<Backend>&) other);
  }
  TypedTensorBase log() const {
    return TensorBase<Backend>::log();
  }
  TensorBase<Backend>& operator+=(const TensorBase<Backend>& rhs) = delete;
  TypedTensorBase& operator+=(const TypedTensorBase& rhs) {
    TensorBase<Backend>::operator+=((const TensorBase<Backend>&)rhs);
    return *this;
  }
  TensorBase<Backend>& operator-=(const TensorBase<Backend>& rhs) = delete;
  TypedTensorBase& operator-=(const TypedTensorBase& rhs) {
    TensorBase<Backend>::operator-=((const TensorBase<Backend>&)rhs);
    return *this;
  }
};

static inline bool can_broadcast_shape(Shape lhs, Shape rhs) {
  if (lhs == rhs) return true;
  size_t min_dim = std::min(lhs.get_ndims(), rhs.get_ndims());
  for (int i = 0; i < min_dim; i++) {
    int64_t& l_dim = lhs[lhs.get_ndims() - i - 1];
    int64_t& r_dim = rhs[rhs.get_ndims() - i - 1];
    if (l_dim != r_dim && l_dim != 1 && r_dim != 1) return false;
  }
  return true;
}

static inline bool can_matmul_shape(Shape lhs, Shape rhs) {
  if (lhs.get_ndims() < 2 || rhs.get_ndims() < 2) return false;
  if (lhs[-1] != rhs[-2]) return false; 
  return can_broadcast_shape(lhs.slice(0, lhs.get_ndims()-2), rhs.slice(0, rhs.get_ndims()-2));
}

static inline bool can_diag_shape(Shape shape) {
  return shape.get_ndims() == 1;
}

static inline Shape calculate_diag_shape(Shape shape) {
  ASSERT(can_diag_shape(shape), "tried to create diag shape with non-compatible type")
  return Shape({shape[0], shape[0]});
}

static inline bool can_broadcast(DataType lhs, DataType rhs) {
  if (lhs.get_element_type() != rhs.get_element_type()) return false;
  return can_broadcast_shape(lhs.get_shape(), rhs.get_shape());
}

static inline bool can_matmul(DataType lhs, DataType rhs) {
  if (lhs.get_element_type() != rhs.get_element_type()) return false;
  return can_matmul_shape(lhs.get_shape(), rhs.get_shape());
}

static inline Shape calculate_broadcast_shape(Shape lhs, Shape rhs) {
  ASSERT(can_broadcast_shape(lhs, rhs), "tried to create broadcast shape with non-compatible types")
  if (lhs == rhs) return lhs;
  auto ldim = lhs.get_ndims(), rdim = rhs.get_ndims();
  auto [min_dim, max_dim] = std::minmax(ldim, rdim);
  Shape res(max_dim);
  for (int i = 0; i < min_dim; i++) {
    int64_t& l_dim = lhs[lhs.get_ndims() - i - 1];
    int64_t& r_dim = rhs[rhs.get_ndims() - i - 1];
    int64_t& res_dim = res[max_dim - i - 1];
    if (l_dim == r_dim) {
      res_dim = l_dim;
      continue;
    }
    if (l_dim == 1)
      res_dim = r_dim;
    else if (r_dim == 1) 
      res_dim = l_dim;
    else {
      panic("broadcast shape checked but it failed to create one");
    }
  }
  Shape larger = lhs.get_ndims() == max_dim ? lhs : rhs;
  for (int i = min_dim; i < max_dim; i++) 
    res[max_dim - i - 1] = larger[max_dim - i - 1];
  return res;
}

static inline Shape calculate_matmul_shape(Shape lhs, Shape rhs) {
  ASSERT(can_matmul_shape(lhs, rhs), "tried to create matmul shape with non-compatible types")
  Shape left = calculate_broadcast_shape(lhs.slice(0,lhs.get_ndims()-2), rhs.slice(0,rhs.get_ndims()-2));
  Shape right({lhs[-2], rhs[-1]});
  return left.concat(right);
}

static inline bool contains_no_more_than_one_any(Shape shape) {
  int cnt = 0;
  for (int i=0;i<shape.get_ndims();i++){
    if (shape[i] == Shape::Any)
      cnt ++;
  }
  return cnt <= 1;
}

static inline bool can_reshape(Shape old, Shape newi) {
  if (!contains_no_more_than_one_any(newi))
    return false;
  bool contains_any = std::any_of(newi.begin(), newi.end(), [](int64_t x) {
    return x == Shape::Any;
  });
  size_t total = old.get_total();
  size_t total_new = newi.get_total_without_any();
  if (contains_any && total_new > total) return false;
  if (!contains_any && total != total_new) return false;
  return true;
}

static inline Shape calculate_reshape_shape(Shape old, Shape newi) {
  size_t total = old.get_total();
  size_t total_new = newi.get_total_without_any();
   for (int i=0;i<newi.get_ndims();i++)
      if (newi[i] == Shape::Any) {
        newi[i] = total/total_new;
        break;
      }
  return newi;
}

template <typename T>
concept IExecutor = requires(T v, typename T::HandleView h, typename T::Handle eh, TensorDescriptor desc, AxisArray axis, Constant val, bool& res_bool) {
  {v.relase(eh)};
  {v.allocate(desc)} -> std::same_as<typename T::Handle>;
  {v.get_data(eh)} -> std::same_as<void*>;
  {v.add(h, h, h)} -> std::same_as<bool>;
  {v.sub(h, h, h)} -> std::same_as<bool>;
  {v.mul(h, h, h)} -> std::same_as<bool>;
  {v.div(h, h, h)} -> std::same_as<bool>;
  {v.max(h, h, h)} -> std::same_as<bool>;
  {v.min(h, h, h)} -> std::same_as<bool>;
  {v.less(h, h, h)} -> std::same_as<bool>;
  {v.less_eq(h, h, h)} -> std::same_as<bool>;
  {v.matmul(h, h, h)} -> std::same_as<bool>;
  {v.sum(h, h, h)} -> std::same_as<bool>;
  {v.mean(h, h, h, axis)} -> std::same_as<bool>;
  {v.reduce_max(h, h, h)} -> std::same_as<bool>;
  {v.reduce_min(h, h, h)} -> std::same_as<bool>;
  {v.copy(h, h)} -> std::same_as<bool>;
  {v.diag(h, h)} -> std::same_as<bool>;
  {v.fill(h, val)} -> std::same_as<bool>;
  {v.equals(res_bool, h, h)} -> std::same_as<bool>;
  {v.near_equals(res_bool, h, h)} -> std::same_as<bool>;
  {v.clip(h, h, val, val)} -> std::same_as<bool>;
  {v.log(h, h)} -> std::same_as<bool>;
  {v.exp(h, h)} -> std::same_as<bool>;
  {v.neg(h, h)} -> std::same_as<bool>;
};


static std::random_device rd;
static std::mt19937 rng(rd());

template<IExecutor Executor>
class Backend {
using ExecutorHandleView = typename Executor::HandleView;
using ExecutorHandle = typename Executor::Handle;
public:
  Backend(Executor&& executor) 
    : executor(std::move(executor)) {}
  Backend(Backend&&) = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  ~Backend() = default;

  class Handle final {
  public:
    Handle() = default;
    Handle(ExecutorHandle handle, TensorDescriptor descriptor) : 
      handle(handle), element_type(descriptor.get_element_type()), 
      shape(descriptor.get_shape()), strides(descriptor.get_strides()), empty(false) {}
    Handle(ExecutorHandle handle, ElementType element_type, Shape shape, Strides strides) :
      handle(handle),  element_type(element_type), shape(shape), strides(strides), empty(false) {}
    size_t get_ndims() const { return shape.get_ndims(); }
    DataType get_datatype() const { return DataType(element_type, shape); }
    ElementType get_element_type() const { return element_type; }
    Shape get_shape() const { return shape; }
    Strides get_strides() const { return strides; }
    TensorDescriptor get_descriptor() const { return TensorDescriptor(element_type, shape, strides); }
    ExecutorHandle get_ehandle() const { return handle; }
    ExecutorHandleView view() const { return ExecutorHandleView{handle, offset, shape, strides}; }
    operator bool() const {
      return !empty;
    }
    friend class Backend;
  private:
    ExecutorHandle handle{};
    ElementType element_type{};
    Shape shape{};
    Strides strides{};
    size_t offset{};
    bool empty { true };
  };

  using Tensor = TensorBase<Backend>;
  template<typename T>
  using TypedTensor = TypedTensorBase<Backend, T>;
  friend class TensorBase<Backend>;

  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  template<signed_type... Sz> \
  Tensor zeros_##name(Sz... sz) {\
    return zeros(DataType(ElementType::enum_name, Shape({sz...})));\
  }\
  template<signed_type... Sz> \
  Tensor ones_##name(Sz... sz) {\
    return ones(DataType(ElementType::enum_name, Shape({sz...})));\
  }\
  template<signed_type... Sz> \
  Tensor uniform_##name(Sz... sz) {\
    return uniform(DataType(ElementType::enum_name, Shape({sz...})));\
  }
  #include "element_type.inc"
  #undef ELEMENT_TYPE

  Tensor zeros(DataType datatype) {
    TensorDescriptor descriptor(datatype.get_shape(), datatype.get_element_type());
    return make_tensor(allocate(descriptor));
  }
  
  Tensor ones(DataType datatype) {
    auto res = zeros(datatype);
    res.fill(Constant(1));
    return res;
  }

  template<typename T>
  Tensor tensor(const std::vector<T>& data) {
    Shape shape = Shape({(int64_t)data.size()});
    auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
    for (int i=0;i<shape[0];i++){
      res.template at<T>(i) = data[i];
    }
    return res;
  }

  template<typename T>
  Tensor tensor(const std::vector<std::vector<T>>& data) {
    Shape shape = Shape({(int64_t)data.size(), (int64_t)data[0].size()});
    auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
    for (int i=0;i<shape[0];i++){
      for (int j=0;j<shape[1];j++){
        res.template at<T>(i,j) = data[i][j];
      }
    }
    return res;
  }

  template<typename T>
  Tensor tensor(const std::vector<std::vector<std::vector<T>>>& data) {
    Shape shape = Shape({(int64_t)data.size(), (int64_t)data[0].size(), (int64_t)data[0][0].size()});
    auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
    for (int i=0;i<shape[0];i++){
      for (int j=0;j<shape[1];j++){
        for (int k=0;k<shape[2];k++){
          res.template at<T>(i,j,k) = data[i][j][k];
        }
      }
    }
    return res;
  }

  Tensor tensor(DataType datatype, Constant val) {
    auto res = zeros(datatype);
    res.fill(val);
    return res;
  }

  Tensor constant(ElementType element_type, Constant val) {
    auto res = zeros(DataType(element_type, Shape({1})));
    res.fill(val);
    return res;
  }

  template<typename T>
  Tensor constant(T val) {
    auto res = zeros(DataType(find_element_type(type_wrapper<T>()), Shape({1})));
    res.fill(val);
    return res;
  }

  Tensor uniform(DataType datatype, double mn=0.0, double mx=1.0) {
    TensorDescriptor descriptor(datatype.get_shape(), datatype.get_element_type());
    Handle res = allocate(descriptor);
    call_with_type([&]<typename T>(type_wrapper<T>) {
      IterUtils::WOffsetView view(get_data(res), descriptor);
      size_t total = view.shape[-1];
      auto operation = [](T a, std::pair<double, double> range) {
        std::uniform_real_distribution<> dist(range.first, range.second);
        return static_cast<T>(dist(rng));
      };
      IterUtils::per_element_self<T, std::pair<double, double>, operation>(view, {mn, mx});
    }, datatype.get_element_type());
    return make_tensor(res);
  }

  Tensor normalize(const Tensor& tensor) {
    return tensor / (tensor * tensor).sum();
  }

  Tensor xavier_normalize(const Tensor& tensor) {
    return tensor / constant(tensor.get_element_type(), std::sqrt(tensor.get_shape()[0]));
  }

  template<typename T, signed_type... Sz>
  TypedTensor<T> zeros(Sz... sz) {
    TensorDescriptor descriptor(Shape({sz...}), find_element_type(type_wrapper<T>()));
    return make_tensor(allocate(descriptor));
  }
private:
  Handle add(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("add", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.add(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle sub(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("sub", lhs, rhs);
    Handle res_ =per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.sub(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle mul(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("mul", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.mul(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle div(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("div", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.div(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle max(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("max", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.max(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle min(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("min", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.min(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle less(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("less", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.less(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle less_eq(Handle lhs, Handle rhs) {
    Handle res = allocate_for_bin_op("less_eq", lhs, rhs);
    Handle res_ = per_element_bin_op_prepare(res, lhs, rhs);
    unwrap(executor.less_eq(res_.view(), lhs.view(), rhs.view()));
    return res;
  }
  bool equals(Handle lhs, Handle rhs) {
    if (lhs.shape != rhs.shape) return false;
    bool res;
    unwrap(executor.equals(res, lhs.view(), rhs.view()));
    return res;
  }
  bool near_equals(Handle lhs, Handle rhs) {
    if (lhs.shape != rhs.shape) return false;
    bool res;
    unwrap(executor.near_equals(res, lhs.view(), rhs.view()));
    return res;
  }
  Handle matmul(Handle lhs, Handle rhs) {
    TYPE_CHECK_OR_THROW(can_matmul(lhs.get_datatype(), rhs.get_datatype()), "matmul", lhs.get_datatype(), rhs.get_datatype())
    Handle res = allocate(TensorDescriptor(calculate_matmul_shape(lhs.get_shape(), rhs.get_shape()), lhs.get_element_type()));
    matmul_prepare(lhs, rhs);
    unwrap(executor.matmul(res.view(), lhs.view(), rhs.view()));
    return res;
  }
  Handle clip(Handle val, Constant mn, Constant mx) {
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor.clip(res.view(), val.view(), mn, mx));
    return res;
  }
  Handle log(Handle val) {
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor.log(res.view(), val.view()));
    return res;
  }
  Handle exp(Handle val) {
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor.exp(res.view(), val.view()));
    return res;
  }
  Handle neg(Handle val) {
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor.neg(res.view(), val.view()));
    return res;
  }
  Handle sum(Handle val, AxisArray axis) {
    Handle res = allocate(val.get_descriptor().set_shape(val.get_shape().reduce(axis)));
    Handle res_ = reduce_prepare(res, val, axis);
    unwrap(executor.sum(res.view(), res_.view(), val.view()));
    return res;
  }
  Handle mean(Handle val, AxisArray axis) {
    Handle res = allocate(val.get_descriptor().set_shape(val.get_shape().reduce(axis)));
    Handle res_ = reduce_prepare(res, val, axis);
    unwrap(executor.mean(res.view(), res_.view(), val.view(), axis));
    return res;
  }
  Handle max(Handle val, AxisArray axis) {
    Handle res = allocate(val.get_descriptor().set_shape(val.get_shape().reduce(axis)));
    Handle res_ = reduce_prepare(res, val, axis);
    unwrap(executor.reduce_max(res.view(), res_.view(), val.view()));
    return res;
  }
  Handle min(Handle val, AxisArray axis) {
    Handle res = allocate(val.get_descriptor().set_shape(val.get_shape().reduce(axis)));
    Handle res_ = reduce_prepare(res, val, axis);
    unwrap(executor.reduce_min(res.view(), res_.view(), val.view()));
    return res;
  }
  Tensor make_tensor(Handle handle) { return Tensor(*this, handle); }
  Handle allocate(TensorDescriptor descriptor) {
    ExecutorHandle ehandle = executor.allocate(descriptor);
    return Handle(ehandle, descriptor);
  }
  Handle copy(Handle handle) {
    Handle res = allocate(handle.get_descriptor());
    executor.copy(res.view(), handle.view());
    return res;
  }
  void fill(Handle handle, Constant val) {
    executor.fill(handle.view(), val);
  }
  Handle reshape(Handle handle, Shape shape) {
    TYPE_CHECK_OR_THROW(can_reshape(handle.shape, shape), "reshape", handle.get_datatype(), handle.get_datatype().set_shape(shape))
    shape = calculate_reshape_shape(handle.shape, shape);
    if (handle.strides.is_contiguous(handle.get_datatype())) {
      handle.strides = Strides(shape, handle.get_element_type());
    } else if (handle.strides.is_reverse_contiguous(handle.get_datatype())) {
      handle.strides = Strides(shape, handle.get_element_type()).reverse();
    } else {
      ASSERT_FAIL(false, "tried to reshape uncontiguous strides")
    }
    handle.shape = shape;
    return handle;
  }

  Handle transpose(Handle handle) {
    if (handle.strides.is_contiguous(handle.get_datatype()) || handle.strides.is_reverse_contiguous(handle.get_datatype())) {
      handle.shape = handle.shape.reverse();
      handle.strides = handle.strides.reverse();
    } else {
      ASSERT_FAIL(false, "tried to transpose uncontiguous strides")
    }
    return handle;
  }
  Handle diag(Handle arr) {
    TYPE_CHECK_OR_THROW(can_diag_shape(arr.get_shape()), "diag", arr.get_datatype())
    Handle res = allocate(arr.get_descriptor().set_shape(calculate_diag_shape(arr.get_shape())));
    executor.diag(res.view(), arr.view());
    return res;
  }
  void release(Handle handle) {
    executor.relase(handle.get_ehandle());
  }
  void* get_data(Handle handle) {
    return executor.get_data(handle.get_ehandle());
  }
  Handle allocate_for_bin_op(const char* opname, Handle lhs, Handle rhs) {
    TYPE_CHECK_OR_THROW(can_broadcast(lhs.get_datatype(), rhs.get_datatype()), opname, lhs.get_datatype(), rhs.get_datatype())
    return allocate(TensorDescriptor(calculate_broadcast_shape(lhs.get_shape(), rhs.get_shape()), lhs.get_element_type()));
  }
  void unwrap(bool eval) {
    CHECK_OR_THROW(eval, ExecutorInternalError)
  }

  Handle per_element_bin_op_prepare(Handle res, Handle& lhs, Handle& rhs) {
    Shape lshape = lhs.get_shape(), rshape = rhs.get_shape();
    if (lshape != rshape) {
      auto ldim = lshape.get_ndims(), rdim = rshape.get_ndims();
      auto [min_dim, max_dim] = std::minmax(ldim, rdim);
      for (int i = 0; i < min_dim; i++) {
        if (lshape[lshape.get_ndims() - i - 1] == 1)
          lhs.strides[lshape.get_ndims() - i - 1] = 0;
        if (rshape[rshape.get_ndims() - i - 1] == 1)
          rhs.strides[rshape.get_ndims() - i - 1] = 0;
      }
      Strides& smaller = lhs.strides.get_ndims() < max_dim ? lhs.strides : rhs.strides;
      Strides new_strides(max_dim);
      for (int i = 0; i < min_dim; i++)
        new_strides[max_dim - i - 1] = smaller[min_dim - i - 1];
      smaller = new_strides;
    }
    return res;
  }

  Handle reduce_prepare(Handle res, Handle& val, AxisArray& axis) {
    axis = val.get_shape().normalize_axis(axis);
    Shape rshape = res.get_shape(), vshape = val.get_shape();
    if (rshape != vshape) {
      Strides new_strides(vshape.get_ndims());
      std::vector<bool> ban(val.get_ndims());
      for (int i : axis) ban[i] = true;
      int ptr = 0;
      for (int i=0;i<val.get_ndims();i++) {
        if (!ban[i])
          new_strides[i] = res.strides[ptr++];
      }
      res.strides = new_strides;
    }
    return res;
  }

  void matmul_prepare(Handle& lhs, Handle& rhs) {
    auto ldim = lhs.get_ndims(), rdim = rhs.get_ndims();
    auto [min_dim, max_dim] = std::minmax(ldim, rdim);
    for (int i = 0; i < min_dim-2; i++) {
      if (lhs.shape[lhs.shape.get_ndims() - i - 3] == 1)
        lhs.strides[lhs.shape.get_ndims() - i - 3] = 0;
      if (rhs.shape[rhs.shape.get_ndims() - i - 3] == 1)
        rhs.strides[rhs.shape.get_ndims() - i - 3] = 0;
    }
    Strides& smaller = lhs.strides.get_ndims() < max_dim ? lhs.strides : rhs.strides;
    Strides new_strides(max_dim);
    for (int i = 0; i < min_dim; i++)
      new_strides[max_dim - i - 1] = smaller[min_dim - i - 1];
    smaller = new_strides;
  }

  Executor executor;
};

}
}