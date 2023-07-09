#pragma once
#include <functional>
#include <initializer_list>
#include <optional>
#include <random>
#include <iostream>
#include <memory>

#include "core.hpp"
#include "types.hpp"
#include "iter_utils.hpp"

namespace katoml {
namespace tensor {

static int64_t MAX_TENSOR_LOG_LIMIT = 128;

template <typename T> concept signed_type = std::is_signed_v<T>;

enum class BinOpcode {
  #define BINOP(def_name, func_name) func_name,
  #define BINOP_NO_CONST(def_name, func_name) BINOP(def_name, func_name)
  #define BINOP_NO_CONST_CPP(def_name, func_name) BINOP(def_name, func_name)
  #define BINOP_CPP(def_name, func_name) BINOP(def_name, func_name)
  #include "operators/bin_def.inc"
  #undef BINOP
  #undef BINOP_NO_CONST
  #undef BINOP_NO_CONST_CPP
  #undef BINOP_CPP
};

static inline const char* opcode_to_string(BinOpcode opcode) {
  switch(opcode) {
  #define BINOP(def_name, func_name) case BinOpcode::func_name: return #func_name;
  #define BINOP_NO_CONST(def_name, func_name) BINOP(def_name, func_name)
  #define BINOP_NO_CONST_CPP(def_name, func_name) BINOP(def_name, func_name)
  #define BINOP_CPP(def_name, func_name) BINOP(def_name, func_name)
  #include "operators/bin_def.inc"
  #undef BINOP
  #undef BINOP_NO_CONST
  #undef BINOP_NO_CONST_CPP
  #undef BINOP_CPP
  }
}

enum class UniOpcode {
  #define UNIOP(def_name, func_name) func_name,
  #define UNIOP_CPP(def_name, func_name) UNIOP(def_name, func_name)
  #include "operators/uni_def.inc"
  #undef UNIOP
  #undef UNIOP_CPP
};

static inline const char* opcode_to_string(UniOpcode opcode) {
  switch(opcode) {
  #define UNIOP(def_name, func_name) case UniOpcode::func_name: return #func_name;
  #define UNIOP_CPP(def_name, func_name) UNIOP(def_name, func_name)
  #include "operators/uni_def.inc"
  #undef UNIOP
  #undef UNIOP_CPP
  }
}

enum class ReduceOpcode {
  #define REDUCEOP(def_name, func_name) func_name,
  #include "operators/reduce_def.inc"
  #undef REDUCEOP
};

static inline const char* opcode_to_string(ReduceOpcode opcode) {
  switch(opcode) {
  #define REDUCEOP(def_name, func_name) case ReduceOpcode::func_name: return #func_name;
  #include "operators/reduce_def.inc"
  #undef REDUCEOP
  }
}

enum class SelfOpcode {
  #define SELFOP(def_name, func_name, fallback) func_name,
  #include "operators/self_def.inc"
  #undef SELFOP
};

static inline const char* opcode_to_string(SelfOpcode opcode) {
  switch(opcode) {
  #define SELFOP(def_name, func_name, fallback) case SelfOpcode::func_name: return #func_name;
  #include "operators/self_def.inc"
  #undef SELFOP
  }
}

class ExecutorHandle {
  public:
  using HandleId = uint32_t;
  static constexpr const HandleId null_id = std::numeric_limits<HandleId>::max();

  ExecutorHandle() = default;
  explicit ExecutorHandle(HandleId id)
    : id(id) {
  }

  operator bool() const { return id != null_id; }
  bool operator==(const ExecutorHandle &rhs) const { return id == rhs.id; }
  bool operator!=(const ExecutorHandle &rhs) const { return id != rhs.id; }
  HandleId get_id() const { return id; }
  void reset() { id = null_id; }

  protected:
    HandleId id{ null_id };
};

class Executor {
public:
  struct HandleView {
    ExecutorHandle handle{};
    size_t offset{};
    Shape shape{};
    Strides strides{};
  };

  virtual ~Executor() = default;
  virtual void relase(ExecutorHandle handle) = 0;
  virtual ExecutorHandle allocate(const TensorDescriptor& desc) = 0;
  virtual void* get_data(ExecutorHandle handle) = 0;
  virtual bool is_alive(ExecutorHandle handle) = 0;
  virtual bool binop(BinOpcode opcode, HandleView res, HandleView lhs, HandleView rhs) = 0;
  virtual bool uniop(UniOpcode opcode, HandleView res, HandleView val) = 0;
  virtual bool reduceop(ReduceOpcode opcode, HandleView res, HandleView res_std, HandleView val, const AxisArray& axis) = 0;
  virtual bool selfop(SelfOpcode opcode, HandleView res, HandleView rhs) = 0;
  virtual bool near_equals(bool& res, HandleView lhs, HandleView rhs) = 0;
  virtual bool clip(HandleView res, HandleView val, Constant mn, Constant mx) = 0;
};

static Constant NEAR_EQUAL_EPS = std::numeric_limits<float>::epsilon()*100; 

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

class TensorBuffer {
public:
  TensorBuffer(TensorDescriptor desc, const uint8_t* buffer_start, size_t size) : 
    desc(desc), buffer(buffer_start, buffer_start+size) {
  }
  TensorDescriptor get_descriptor() const {
    return desc;
  }
  const uint8_t* data() const {
    return buffer.data();
  }
  size_t size() const {
    return buffer.size();
  }
private:
  TensorDescriptor desc;
  std::vector<uint8_t> buffer; 
};

static inline std::random_device rd;
static inline std::mt19937 rng(rd());

class Tensor;
template<typename T>
class TypedTensor;

class Backend {
public:
  Backend(std::unique_ptr<Executor>&& executor) 
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
    Executor::HandleView view() const { return Executor::HandleView{handle, offset, shape, strides}; }
    operator bool() const {
      return !empty;
    }
    bool operator==(const Handle& other) const {
      return handle == other.handle;
    }
    bool operator!=(const Handle& other) const {
      return handle != other.handle;
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

  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  template<signed_type... Sz> \
  Tensor zeros_##name(Sz... sz); \
  template<signed_type... Sz> \
  Tensor ones_##name(Sz... sz); \
  template<signed_type... Sz> \
  Tensor uniform_##name(Sz... sz);
  #include "element_type.inc"
  #undef ELEMENT_TYPE

  #define BINOP_NO_CONST(def_name, func_name)\
  inline Tensor def_name(const Tensor& lhs, const Tensor& rhs);
  #define BINOP_NO_CONST_CPP(def_name, func_name) ;
  #define BINOP(def_name, func_name) \
  BINOP_NO_CONST(def_name, func_name) \
  inline Tensor def_name(const Tensor& lhs, Constant rhs); \
  inline Tensor def_name(Constant lhs, const Tensor& rhs);
  #define BINOP_CPP(def_name, func_name) ;
  #define UNIOP(def_name, func_name) inline Tensor def_name(const Tensor& val);
  #define UNIOP_CPP(def_name, func_name) ;
  #define REDUCEOP(def_name, func_name) inline Tensor def_name(const Tensor& val, const AxisArray& axis = AllAxis);
  #define SELFOP(def_name, func_name, fallback) ;
  #include "operators/operators.inc"

  inline Tensor zeros(DataType datatype);
  inline Tensor ones(DataType datatype);
  template<typename T>
  inline Tensor tensor(std::initializer_list<T> data);
  template<typename T>
  inline Tensor tensor(std::initializer_list<std::initializer_list<T>> data);
  template<typename T>
  inline Tensor tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> data);
  template<typename T>
  inline Tensor from_vector(const std::vector<T>& data);
  template<typename T>
  inline Tensor from_vector(const std::vector<std::vector<T>>& data);
  template<typename T>
  inline Tensor from_vector(const std::vector<std::vector<std::vector<T>>>& data);
  inline Tensor tensor(DataType datatype, Constant val);
  inline Tensor constant(ElementType element_type, Constant val);
  template<typename T>
  inline Tensor constant(T val);
  inline Tensor uniform(DataType datatype, double mn=0.0, double mx=1.0);
  inline Tensor rand_normal(DataType datatype, double mean=0.0, double std=1.0);
  inline Tensor normalize(const Tensor& tensor);
  template<typename T, signed_type... Sz>
  inline TypedTensor<T> zeros(Sz... sz);
  inline Tensor load(const TensorBuffer& buffer);

  friend class Tensor;
private:
  std::unique_ptr<Executor> executor;

  Handle binop(BinOpcode opcode, Handle lhs, Handle rhs) {
    if (opcode == BinOpcode::matmul) {
      return matmul(lhs, rhs);
    }
    Handle res = allocate_for_bin_op(opcode_to_string(opcode), lhs, rhs);
    per_element_bin_op_prepare(lhs, rhs);
    unwrap(executor->binop(opcode, res.view(), lhs.view(), rhs.view()));
    return res;
  }

  Handle uniop(UniOpcode opcode, Handle val) {
    if (opcode == UniOpcode::diag)
      return diag(val);
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor->uniop(opcode, res.view(), val.view()));
    return res;
  }

  Handle selfop(SelfOpcode opcode, Handle res, Handle rhs) {
    switch(opcode) {
    #define SELFOP(def_name, func_name, fallback) case SelfOpcode::func_name: {\
      if (!can_no_alloc_bin_assign_op(opcode_to_string(opcode), res, rhs))\
        return binop(BinOpcode::fallback, res, rhs); \
      break;\
    }
    #include "operators/self_def.inc"
    #undef SELFOP
    }
    per_element_bin_op_prepare(res, rhs);
    unwrap(executor->selfop(opcode, res.view(), rhs.view()));   
    return res;
  }

  Handle reduceop(ReduceOpcode opcode, Handle val, AxisArray axis) {
    Handle res = allocate(val.get_descriptor().set_shape(val.get_shape().reduce(axis)));
    Handle res_ = reduce_prepare(res, val, axis);
    unwrap(executor->reduceop(opcode, res.view(), res_.view(), val.view(), axis));
    return res;
  }
  bool near_equals(Handle lhs, Handle rhs) {
    if (lhs.shape != rhs.shape) return false;
    bool res;
    unwrap(executor->near_equals(res, lhs.view(), rhs.view()));
    return res;
  }
  Handle clip(Handle val, Constant mn, Constant mx) {
    Handle res = allocate(TensorDescriptor(val.get_shape(), val.get_element_type()));
    unwrap(executor->clip(res.view(), val.view(), mn, mx));
    return res;
  }

  Handle matmul(Handle lhs, Handle rhs) {
    TYPE_CHECK_OR_THROW(can_matmul(lhs.get_datatype(), rhs.get_datatype()), "matmul", lhs.get_datatype(), rhs.get_datatype())
    Handle res = allocate(TensorDescriptor(calculate_matmul_shape(lhs.get_shape(), rhs.get_shape()), lhs.get_element_type()));
    matmul_prepare(lhs, rhs);
    unwrap(executor->binop(BinOpcode::matmul, res.view(), lhs.view(), rhs.view()));
    return res;
  }
 
  Handle diag(Handle arr) {
    TYPE_CHECK_OR_THROW(can_diag_shape(arr.get_shape()), "diag", arr.get_datatype())
    Handle res = allocate(arr.get_descriptor().set_shape(calculate_diag_shape(arr.get_shape())));
    executor->uniop(UniOpcode::diag, res.view(), arr.view());
    return res;
  }

  inline Tensor make_tensor(Handle handle);
  Handle allocate(TensorDescriptor descriptor) {
    ExecutorHandle ehandle = executor->allocate(descriptor);
    return Handle(ehandle, descriptor);
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

  void release(Handle handle) {
    executor->relase(handle.get_ehandle());
  }
  void* get_data(Handle handle) {
    return executor->get_data(handle.get_ehandle());
  }
  bool is_alive(Handle handle) {
    return executor->is_alive(handle.get_ehandle());
  }
  Handle allocate_for_bin_op(const char* opname, Handle lhs, Handle rhs) {
    TYPE_CHECK_OR_THROW(can_broadcast(lhs.get_datatype(), rhs.get_datatype()), opname, lhs.get_datatype(), rhs.get_datatype())
    return allocate(TensorDescriptor(calculate_broadcast_shape(lhs.get_shape(), rhs.get_shape()), lhs.get_element_type()));
  }
  void unwrap(bool eval) {
    CHECK_OR_THROW(eval, ExecutorInternalError)
  }
  void per_element_bin_op_prepare(Handle& lhs, Handle& rhs) {
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
  }
  bool can_no_alloc_bin_assign_op(const std::string& opname, Handle res, Handle rhs) {
    TYPE_CHECK_OR_THROW(can_broadcast(res.get_datatype(), rhs.get_datatype()), opname, res.get_datatype(), rhs.get_datatype())
    return res.get_shape() == calculate_broadcast_shape(res.get_shape(), rhs.get_shape());
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

};

class Tensor {
public:
  Tensor(Backend& backend, Backend::Handle handle, bool is_view = false) : backend(backend), handle(handle), is_view(is_view) {}
  Tensor(const Tensor& other) = delete;
  Tensor& operator=(const Tensor& other) = delete;
  Tensor(Tensor&& other) : backend(other.backend), handle(other.handle), is_view(other.is_view) {
    other.handle = {};
  }
  Tensor& operator=(Tensor&& other) {
    std::swap(backend, other.backend);
    std::swap(handle, other.handle);
    std::swap(is_view, other.is_view);
    return *this;
  }
  ~Tensor() {
    if (handle && !is_view) {
      backend.get().release(handle);
    }
  }
  int get_ndims() const { return handle.get_ndims(); }
  ElementType get_element_type() const { return handle.get_element_type(); }
  Shape get_shape() const { return handle.get_shape(); }
  Strides get_strides() const { return handle.get_strides(); }
  DataType get_datatype() const { return DataType(get_element_type(), get_shape()); }
  TensorDescriptor get_descriptor() const { 
    return TensorDescriptor(handle.get_element_type(), handle.get_shape(), handle.get_strides()); 
  }
  template<signed_type... Index>
  Constant at(Index... index) const {
    return call_with_type<Constant>([&]<typename T>(type_wrapper<T>) {
      return *reinterpret_cast<const T*>(get_data() + calculate_offset(index...));
    }, get_element_type());
  }
  template<signed_type... Index>
  Constant operator()(Index... index) const {
    return at(index...);
  }
  using SlowTensorIndex = std::vector<int>;
  Constant at_slow(const SlowTensorIndex& index) const {
    return call_with_type<Constant>([&]<typename T>(type_wrapper<T>) {
      return *reinterpret_cast<const T*>(get_data() + calculate_offset(index));
    }, get_element_type());
  }
  class ConstantSetter {
  public:
    ConstantSetter& operator=(Constant constant) {
      call_with_type<void>([&]<typename T>(type_wrapper<T>) {
        *reinterpret_cast<T*>(parent.get().get_data() + offset) = constant.cast<T>();
      }, parent.get().get_element_type());
      return *this;
    }
    template<typename T>
    T cast() {
      return call_with_type<T>([&]<typename F>(type_wrapper<F>) {
        return static_cast<T>(*reinterpret_cast<F*>(parent.get().get_data() + offset));
      }, parent.get().get_element_type());
    }
    template<typename T>
    T& raw() {
      return *reinterpret_cast<T*>(parent.get().get_data() + offset);
    }
    friend class Tensor;
  private:
    ConstantSetter(Tensor& parent, size_t offset) : parent(parent), offset(offset) {}
    std::reference_wrapper<Tensor> parent;
    size_t offset;
  };
  template<signed_type... Index>
  ConstantSetter at(Index... index) {
    return ConstantSetter(*this, calculate_offset(index...));
  }
  ConstantSetter at_slow(const SlowTensorIndex& index) {
    return ConstantSetter(*this, calculate_offset(index));
  }
  template<signed_type... Index>
  ConstantSetter operator()(Index... index) {
    return at(index...);
  }
  template<typename T, signed_type... Index>
  T at_typed(Index... index) const {
    return at(index...).template cast<T>();
  }
  template<typename T, signed_type... Index>
  T& at_typed(Index... index) {
    return at(index...).template raw<T>();
  }

  #define BINOP_NO_CONST(def_name, func_name)\
  Tensor def_name(const Tensor& rhs) const {\
    sanity_check(rhs);\
    return Tensor(backend.get(), backend.get().binop(BinOpcode::func_name, handle, rhs.handle));\
  }
  #define BINOP_NO_CONST_CPP(def_name, func_name) BINOP_NO_CONST(def_name, func_name)
  #define BINOP(def_name, func_name) \
  BINOP_NO_CONST(def_name, func_name) \
  Tensor def_name(Constant rhs) const {\
    sanity_check();\
    auto constant = backend.get().constant(get_element_type(), rhs);\
    return def_name(constant);\
  }
  #define BINOP_CPP(def_name, func_name) \
  BINOP(def_name, func_name)\
  friend Tensor def_name(Constant lhs, const Tensor& rhs) {\
    auto constant = rhs.backend.get().constant(rhs.get_element_type(), lhs);\
    return constant.def_name(rhs);\
  }
  #define UNIOP(def_name, func_name) \
  Tensor def_name() const {\
    sanity_check();\
    return Tensor(backend.get(), backend.get().uniop(UniOpcode::func_name, handle));\
  }
  #define UNIOP_CPP(def_name, func_name) UNIOP(def_name, func_name)
  #define REDUCEOP(def_name, func_name) \
  Tensor def_name(const AxisArray& axis = AllAxis) const {\
    sanity_check();\
    return Tensor(backend.get(), backend.get().reduceop(ReduceOpcode::func_name, handle, axis));\
  }
  #define SELFOP(def_name, func_name, fallback) \
  Tensor& def_name(const Tensor& rhs) {\
    sanity_check(rhs);\
    assign_handle(backend.get().selfop(SelfOpcode::func_name, handle, rhs.handle));\
    return *this;\
  }\
  Tensor& def_name(Constant rhs) {\
    sanity_check();\
    auto constant = backend.get().constant(get_element_type(), rhs);\
    return def_name(constant);\
  }
  #include "operators/operators.inc"

  bool near_equals(const Tensor& rhs) const {
    sanity_check(rhs);
    return backend.get().near_equals(handle, rhs.handle);
  }
  Tensor clip(Constant mn, Constant mx) const {
    sanity_check();
    return Tensor(backend.get(), backend.get().clip(handle, mn, mx));
  }
  
  // ==========================================
  // * operations that can return tensor view *
  Tensor reshaped(Shape shape) const {
    sanity_check();
    return wrap(backend.get().reshape(handle, shape));
  }
  Tensor transposed() const {
    sanity_check();
    return wrap(backend.get().transpose(handle));
  }
  Tensor axis_extended() const {
    sanity_check();
    return reshaped(get_shape().extend_axis());
  }
  // ==========================================

  // =============================
  // * self modifying operations *
  void reshape(Shape shape) {
    sanity_check();
    assign_handle(backend.get().reshape(handle, shape));
  }
  void transpose() {
    assign_handle(backend.get().transpose(handle));
  }
  void extend_axis() {
    sanity_check();
    reshape(get_shape().extend_axis());
  }
  // =============================

  TensorBuffer save() const {
    return TensorBuffer(get_descriptor(), get_data(), get_descriptor().get_data_size());
  }

  using IterateFunc = std::function<void(const SlowTensorIndex& index)>;
  void iterate_slow(IterateFunc&& func) const {
    auto dfs = [&](auto self, SlowTensorIndex cur) -> void {
      if (cur.size() == get_ndims()) {
        func(cur);
        return;
      }
      for (uint64_t i=0;i<get_shape()[cur.size()];i++){
        cur.push_back(i);
        self(self, cur);
        cur.pop_back();
      }
    };
    dfs(dfs, {});
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    std::ostringstream ss("");
    call_with_type<void>([&]<typename T>(type_wrapper<T>) {
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

  Backend& get_backed() const {
    return backend.get();
  }
protected:
  uint8_t* get_data() {
    return reinterpret_cast<uint8_t*>(backend.get().get_data(handle));
  }
  const uint8_t* get_data() const {
    return reinterpret_cast<const uint8_t*>(backend.get().get_data(handle));
  }
  Tensor wrap(Backend::Handle newi) const {
    if (handle == newi) 
      return Tensor(backend.get(), newi, true);
    return Tensor(backend.get(), newi);
  }
  void assign_handle(Backend::Handle newi) {
    CHECK_OR_THROW(!is_view || handle == newi, ViewAssignAllocationError)
    if (handle != newi) {
      backend.get().release(handle);
    }
    handle = newi;
  }
  void sanity_check() const {
    if (is_view)
      CHECK_OR_THROW(backend.get().is_alive(handle), UseAfterFreeError)
  }
  void sanity_check(const Tensor& other) const {
    CHECK_OR_THROW(&backend.get() == &other.backend.get(), BackendMismatchError)
    if (is_view)
      CHECK_OR_THROW(backend.get().is_alive(handle), UseAfterFreeError)
    if (other.is_view)
      CHECK_OR_THROW(backend.get().is_alive(other.handle), UseAfterFreeError)
  }
  template<signed_type... Index>
  size_t calculate_offset(Index... index) const {
    int i = 0;
    size_t offset = 0;
    ((offset += index * handle.get_strides()[i++]), ...);
    ASSERT(i == get_ndims(), "tensor at index not matching dimension");
    return offset;
  }
  size_t calculate_offset(const SlowTensorIndex& index) const {
    ASSERT(index.size() == get_ndims(), "tensor at index not matching dimension");
    size_t offset = 0;
    for (int i=0;i<get_ndims();i++){
      offset += index[i] * handle.get_strides()[i];
    }
    return offset;
  }
  std::reference_wrapper<Backend> backend;
  bool is_view;
  Backend::Handle handle;
};

template<typename T>
class TypedTensor : public Tensor {
public:
  TypedTensor() = default;
  TypedTensor(Tensor&& other) : Tensor(std::move(other)) {
    CHECK_OR_THROW(find_element_type(type_wrapper<T>()) == this->get_element_type(), InvalidTypedError)
  }
  TypedTensor(TypedTensor&& other) : Tensor(std::move(other)) {
    CHECK_OR_THROW(find_element_type(type_wrapper<T>()) == this->get_element_type(), InvalidTypedError)
  }
  TypedTensor(const TypedTensor& other) = delete;
  TypedTensor& operator=(const TypedTensor& other) = delete;
  TypedTensor& operator=(Tensor&& other) {
    Tensor::operator=(std::move(other));
    return *this;
  }
  TypedTensor& operator=(TypedTensor&& other) {
    Tensor::operator=(std::move(other));
    return *this;
  }
  template<signed_type... Index>
  Constant at(Index... index) const = delete;
  template<signed_type... Index>
  ConstantSetter at(Index...index) = delete;
  template<typename F, signed_type... Index>
  F at_typed(Index... index) const = delete;
  template<typename F, signed_type... Index>
  F& at_typed(Index... index) = delete;

  template<signed_type... Index>
  T at(Index... index) const { return Tensor::at(index...).template cast<T>(); }
  template<signed_type... Index>
  T& at(Index...index) { return Tensor::at(index...).template cast<T>(); }
  // FIXME
  // template<signed_type... Index>
  // T operator()(Index... index) const { return Tensor::at(index...).template cast<T>(); }
  // template<signed_type... Index>
  // T& operator()(Index...index) { return Tensor::at(index...).template cast<T>(); }

  #define BINOP_NO_CONST(def_name, func_name)\
  Tensor def_name(const Tensor& rhs) const = delete;\
  TypedTensor def_name(const TypedTensor& rhs) const {\
    return Tensor::def_name(rhs);\
  }
  #define BINOP(def_name, func_name) \
  BINOP_NO_CONST(def_name, func_name) \
  TypedTensor def_name(Constant rhs) const {\
    return Tensor::def_name(rhs);\
  }
  #define BINOP_NO_CONST_CPP(def_name, func_name) BINOP_NO_CONST(def_name, func_name)
  #define BINOP_CPP(def_name, func_name) \
  BINOP(def_name, func_name) 
  // FIXME
  // friend TypedTensor def_name(Constant lhs, const TypedTensor& rhs) {\
  //   return Tensor::def_name(lhs, rhs);\
  // }
  #define UNIOP(def_name, func_name) \
  TypedTensor def_name() const {\
    return Tensor::def_name();\
  }
  #define UNIOP_CPP(def_name, func_name) UNIOP(def_name, func_name)
  #define REDUCEOP(def_name, func_name) \
  TypedTensor def_name(const AxisArray& axis = AllAxis) const {\
    return Tensor::def_name(axis);\
  }
  #define SELFOP(def_name, func_name, fallback) \
  TypedTensor& def_name(const TypedTensor& rhs) {\
    return Tensor::def_name(rhs);\
  }\
  TypedTensor& def_name(Constant rhs) {\
    return Tensor::def_name(rhs);\
  }
  #include "operators/operators.inc"

  bool near_equals(const Tensor& rhs) const = delete;
  bool near_equals(const TypedTensor& rhs) const {
    return Tensor::near_equals(rhs);
  }
  TypedTensor clip(Constant mn, Constant mx) const {
    return Tensor::clip(mn, mx);
  }
  
  // ==========================================
  // * operations that can return tensor view *
  TypedTensor reshaped(Shape shape) const {
    return Tensor::reshaped(shape);
  }
  TypedTensor transposed() const {
    return Tensor::transposed();
  }
  TypedTensor axis_extended() const {
    return Tensor::axis_extended();
  }
  // ==========================================
};


#define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
template<signed_type... Sz> \
Tensor Backend::zeros_##name(Sz... sz) {\
  return zeros(DataType(ElementType::enum_name, Shape({sz...})));\
}\
template<signed_type... Sz> \
Tensor Backend::ones_##name(Sz... sz) {\
  return ones(DataType(ElementType::enum_name, Shape({sz...})));\
}\
template<signed_type... Sz> \
Tensor Backend::uniform_##name(Sz... sz) {\
  return uniform(DataType(ElementType::enum_name, Shape({sz...})));\
}
#include "element_type.inc"
#undef ELEMENT_TYPE

#define BINOP_NO_CONST(def_name, func_name)\
Tensor Backend::def_name(const Tensor& lhs, const Tensor& rhs) {\
  return lhs.def_name(rhs);\
}
#define BINOP_NO_CONST_CPP(def_name, func_name) ;
#define BINOP(def_name, func_name) \
BINOP_NO_CONST(def_name, func_name) \
Tensor Backend::def_name(const Tensor& lhs, Constant rhs) {\
  return lhs.def_name(rhs);\
}\
Tensor Backend::def_name(Constant lhs, const Tensor& rhs) {\
  auto c = constant(rhs.get_element_type(), lhs);\
  return c.def_name(rhs);\
}
#define BINOP_CPP(def_name, func_name) ;
#define UNIOP(def_name, func_name) \
Tensor Backend::def_name(const Tensor& val) {\
  return val.def_name();\
}
#define UNIOP_CPP(def_name, func_name) ;
#define REDUCEOP(def_name, func_name) \
Tensor Backend::def_name(const Tensor& val, const AxisArray& axis) {\
  return val.def_name(axis);\
}
#define SELFOP(def_name, func_name, fallback) ;
#include "operators/operators.inc"

Tensor Backend::zeros(DataType datatype) {
  TensorDescriptor descriptor(datatype.get_shape(), datatype.get_element_type());
  return make_tensor(allocate(descriptor));
}

Tensor Backend::ones(DataType datatype) {
  auto res = zeros(datatype);
  res.assign(1);
  return res;
}

template<typename T>
Tensor Backend::tensor(std::initializer_list<T> data) {
  Shape shape = Shape({(int64_t)data.size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0; auto x : data){
    res.at<T>(i) = x;
    i++;
  }
  return res;
}

template<typename T>
Tensor Backend::tensor(std::initializer_list<std::initializer_list<T>> data) {
  Shape shape = Shape({(int64_t)data.size(), (int64_t)data.begin()->size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0; auto& arr0 : data){
    for (int j=0; auto x : arr0){
      res.at<T>(i,j) = x;
      j++;
    }
    i++;
  }
  return res;
}

template<typename T>
Tensor Backend::tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> data) {
  Shape shape = Shape({(int64_t)data.size(), (int64_t)data.begin()->size(), (int64_t)data.begin()->begin()->size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0; auto& arr0 : data){
    for (int j=0; auto& arr1 : arr0){
      for (int k=0; auto x : arr1){
        res.at<T>(i,j,k) = x;
        k++;
      }
      j++;
    }
    i++;
  }
  return res;
}

template<typename T>
Tensor Backend::from_vector(const std::vector<T>& data) {
  Shape shape = Shape({(int64_t)data.size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0;i<shape[0];i++){
    res.at<T>(i) = data[i];
  }
  return res;
}

template<typename T>
Tensor Backend::from_vector(const std::vector<std::vector<T>>& data) {
  Shape shape = Shape({(int64_t)data.size(), (int64_t)data[0].size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0;i<shape[0];i++){
    for (int j=0;j<shape[1];j++){
      res.at<T>(i,j) = data[i][j];
    }
  }
  return res;
}

template<typename T>
Tensor Backend::from_vector(const std::vector<std::vector<std::vector<T>>>& data) {
  Shape shape = Shape({(int64_t)data.size(), (int64_t)data[0].size(), (int64_t)data[0][0].size()});
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), shape));
  for (int i=0;i<shape[0];i++){
    for (int j=0;j<shape[1];j++){
      for (int k=0;k<shape[2];k++){
        res.at<T>(i,j,k) = data[i][j][k];
      }
    }
  }
  return res;
}

Tensor Backend::tensor(DataType datatype, Constant val) {
  auto res = zeros(datatype);
  res.at(0) = val;
  return res;
}

Tensor Backend::constant(ElementType element_type, Constant val) {
  auto res = zeros(DataType(element_type, Shape({1})));
  res.at(0) = val;
  return res;
}

template<typename T>
Tensor Backend::constant(T val) {
  auto res = zeros(DataType(find_element_type(type_wrapper<T>()), Shape({1})));
  res.at(0).cast<T>() = val;
  return res;
}

// FIXME: maybe let backend do this?
Tensor Backend::uniform(DataType datatype, double mn, double mx) {
  TensorDescriptor descriptor(datatype.get_shape(), datatype.get_element_type());
  Handle res = allocate(descriptor);
  call_with_type<void>([&]<typename T>(type_wrapper<T>) {
    IterUtils::WOffsetView view(get_data(res), descriptor);
    auto operation = [](T a, std::pair<double, double> range) {
      std::uniform_real_distribution<> dist(range.first, range.second);
      return static_cast<T>(dist(rng));
    };
    IterUtils::per_element_self<T, std::pair<double, double>, operation>(view, {mn, mx});
  }, datatype.get_element_type());
  return make_tensor(res);
}

Tensor Backend::rand_normal(DataType datatype, double mean, double std) {
  TensorDescriptor descriptor(datatype.get_shape(), datatype.get_element_type());
  Handle res = allocate(descriptor);
  call_with_type<void>([&]<typename T>(type_wrapper<T>) {
    IterUtils::WOffsetView view(get_data(res), descriptor);
    auto operation = [](T a, std::pair<double, double> range) {
      std::normal_distribution<> dist(range.first, range.second);
      return static_cast<T>(dist(rng));
    };
    IterUtils::per_element_self<T, std::pair<double, double>, operation>(view, {mean, std});
  }, datatype.get_element_type());
  return make_tensor(res);
}

Tensor Backend::normalize(const Tensor& tensor) {
  return tensor / (tensor * tensor).sum();
}

template<typename T, signed_type... Sz>
TypedTensor<T> Backend::zeros(Sz... sz) {
  TensorDescriptor descriptor(Shape({sz...}), find_element_type(type_wrapper<T>()));
  return make_tensor(allocate(descriptor));
}

Tensor Backend::make_tensor(Handle handle) { 
  return Tensor(*this, handle); 
}

Tensor Backend::load(const TensorBuffer& buffer) {
  auto handle = allocate(buffer.get_descriptor());
  uint8_t* data = reinterpret_cast<uint8_t*>(get_data(handle));
  std::memcpy(data, buffer.data(), buffer.size());
  return make_tensor(handle);
}

}
}