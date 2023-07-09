#pragma once
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>
#include <variant>
#include <atomic>
#include <type_traits>
#include <vector>
#include <array>
#include "../mlsupport/errors.hpp"

namespace katoml {
namespace tensor {

constexpr static int MAX_DIM = 4;

enum class ElementType {
  None,
#define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) enum_name,
#include "element_type.inc"
#undef ELEMENT_TYPE
};

inline std::ostream& operator<<(std::ostream& os, ElementType type) {
  switch (type) {
#define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  case ElementType::enum_name:\
    return os << #enum_name;
#include "element_type.inc"
#undef ELEMENT_TYPE
  case ElementType::None: 
    return os << "None";
  }
}

inline size_t get_element_type_size(ElementType type) {
  switch (type) {
  case ElementType::None:
    return 0;
  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  case ElementType::enum_name:\
    return bytes_size;
  #include "element_type.inc"
  #undef ELEMENT_TYPE
  }
}

using AxisArray = std::vector<int>;
const static AxisArray AllAxis = {MAX_DIM};

class Shape {
using Array = std::array<int64_t, MAX_DIM>;
public:
  static constexpr const int64_t Any = 0;
  Shape() = default;
  Shape(size_t ndims) : ndims(ndims) {
    ASSERT(ndims <= MAX_DIM, "shape maximum dimension reached")
  }
  Shape(std::initializer_list<int64_t> list) :
    ndims(list.size()) {
    ASSERT(list.size() <= MAX_DIM, "shape maximum dimension reached")
    int i = 0;
    for (int64_t v : list) {
      num_elements[i++] = v;
    }
  }
  Shape(size_t ndims, Array num_elements) : 
    ndims(ndims), num_elements(num_elements) {}
  const int64_t* begin() const {
    return &num_elements[0];
  }
  const int64_t* end() const {
    return &num_elements[ndims];
  }
  int64_t* begin() {
    return &num_elements[0];
  }
  int64_t* end() {
    return &num_elements[ndims];
  }
  size_t get_ndims() const { return ndims; }
  Array get_num_elements() const { return num_elements; }
  size_t get_total() const { 
    size_t res = 1;
    for (int i=0;i<ndims;i++){
      res *= num_elements[i];
    }
    return res;
  }
  size_t get_total_without_any() const { 
    size_t res = 1;
    for (int i=0;i<ndims;i++){
      if (num_elements[i] != Any)
        res *= num_elements[i];
    }
    return res;
  }
  int64_t operator[](int i) const { 
    if (i >= 0) {
      ASSERT(i < ndims, "shape out of bound")
      return num_elements[i];
    } else {
      ASSERT(i+ndims >= 0, "shape out of bound")
      return num_elements[i+ndims];
    }
  }
  int64_t& operator[](int i) { 
    if (i >= 0) {
      ASSERT(i < ndims, "shape out fo bound")
      return num_elements[i];
    } else {
      ASSERT(i+ndims >= 0, "shape out of bound")
      return num_elements[i+ndims];
    }
  }
  Shape slice(int s, int e) const {
    ASSERT(s >= 0 && s < ndims, "shape slice out of bound")
    ASSERT(e >= 0 && e < ndims, "shape slice out of bound")
    ASSERT(s <= e, "shape slice start not less than or equal to end")
    Shape res(e-s);
    for (int i=s;i<e;i++) {
      res[i] = num_elements[i];
    }
    return res;
  }
  Shape concat(Shape other) const {
    Shape res(ndims + other.ndims);
    for (int i=0;i<ndims;i++) res[i] = num_elements[i];
    for (int i=0;i<other.ndims;i++) res[i+ndims] = other.num_elements[i];
    return res;
  }
  Shape insert_axis(int axis) const {
    Shape res(ndims+1);
    if (axis < 0) {
      axis += ndims;
    }
    ASSERT(axis >= 0 && axis <= ndims, "shape axis out of bound")
    for (int i=0;i<ndims;i++){
      if (i < axis)
        res[i] = num_elements[i];
      else
        res[i+1] = num_elements[i];
    }
    res[axis] = 1;
    return res;
  }
  Shape extend_axis() const {
    return insert_axis(ndims);
  }
  Shape reverse() const {
    Shape res = *this;
    std::reverse(std::begin(res.num_elements), std::begin(res.num_elements)+ndims);
    return res;
  }
  Shape reduce(AxisArray axis) const {
    if (axis.size() == 0) {
      return *this;
    }
    axis = normalize_axis(axis);
    CHECK_OR_THROW(std::all_of(std::begin(axis), std::end(axis), [&](int i) {
      return i >= 0 && i < ndims;
    }), InvalidReduceAxisError)
    std::vector<bool> ban(ndims);
    for (int i : axis) ban[i] = true;
    int ptr = 0;
    Shape res(ndims-axis.size());
    for (int i=0;i<ndims;i++){
      if (!ban[i])
        res[ptr++] = num_elements[i];
    }
    if (res.get_ndims() == 0)
      return Shape({1});
    return res;
  }
  AxisArray normalize_axis(AxisArray axis) const {
    if (axis == AllAxis) {
      axis.assign(ndims, 0);
      std::iota(std::begin(axis), std::end(axis), 0);
      return axis;
    }
    for (int& i : axis) {
      if (i < 0) i += ndims;
    }
    return axis;
  }
  bool compatible(const Shape& other) const {
    if (get_ndims() != other.get_ndims()) return false;
    for (int i=0;i<get_ndims();i++){
      if (num_elements[i] != Any && 
        other.num_elements[i] != Any && 
        num_elements[i] != other.num_elements[i]) {
        return false;
      }
    }
    return true;
  }
  bool operator==(const Shape& other) const {
    if (ndims != other.ndims) return false;
    for (int i=0;i<ndims;i++)
      if (num_elements[i] != other.num_elements[i])
        return false;
    return true;
  }
  bool operator!=(const Shape& other) const {
    return !(*this == other);
  }

private:
  size_t ndims{};
  Array num_elements{};
};

inline std::ostream& operator<<(std::ostream& os, Shape shape) {
  os << "[";
  for (int i=0;i<shape.get_ndims();i++){
    if (i != shape.get_ndims()-1)
      os << shape[i] << ", ";
    else
      os << shape[i];
  }
  return os << "]";
}

static inline size_t calculate_reduced_count(Shape shape, AxisArray axis){
  axis = shape.normalize_axis(axis);
  size_t cnt = 1;
  for (int i : axis) {
    cnt *= shape[i];
  }
  return cnt;
}

class DataType {
public:
  DataType() = default;
  DataType(ElementType element_type, Shape shape) : 
    element_type(element_type), shape(shape) {}
  ElementType get_element_type() const { return element_type; }
  Shape get_shape() const { return shape; }
  DataType set_shape(Shape shape) {
    DataType res = *this;
    res.shape = shape;
    return res;
  }
private:
  ElementType element_type{};
  Shape shape{};
};

class Strides {
using Array = std::array<size_t, MAX_DIM>;
public:
  Strides() = default;
  Strides(size_t ndims) : ndims(ndims) {
    ASSERT(ndims <= MAX_DIM, "strides maximum dimension reached")
  }
  Strides(size_t ndims, Array num_bytes) : 
    ndims(ndims), num_bytes(num_bytes) {
    ASSERT(ndims <= MAX_DIM, "strides maximum dimension reached")
  }
  Strides(Shape shape, ElementType element_type) 
    : ndims(shape.get_ndims()) {
    ASSERT(ndims <= MAX_DIM, "strides maximum dimension reached")
    num_bytes[ndims-1] = get_element_type_size(element_type);
    for (int i=(int)ndims-2;i>=0;i--){
      num_bytes[i] = num_bytes[i+1] * shape[i+1];
    }
  }
  size_t get_ndims() const { return ndims; }
  Array get_num_bytes() const { return num_bytes; }
  size_t operator[](int i) const { 
    if (i >= 0) {
      ASSERT(i < ndims, "strides out of bound")
      return num_bytes[i];
    } else {
      ASSERT(i+ndims >= 0, "strides out of bound")
      return num_bytes[i+ndims];
    }
  }
  size_t& operator[](int i) { 
    if (i >= 0) {
      ASSERT(i < ndims, "strides out of bound")
      return num_bytes[i];
    } else {
      ASSERT(i+ndims >= 0, "strides out of bound")
      return num_bytes[i+ndims];
    }
  }
  Strides slice(int s, int e) const {
    ASSERT(s >= 0 && s < ndims, "strides slice out of bound")
    ASSERT(e >= 0 && e < ndims, "strides slice out of bound")
    ASSERT(s <= e, "strides slice start not less than or equal to end")
    Strides res(e-s);
    for (int i=s;i<e;i++) {
      res[i] = num_bytes[i];
    }
    return res;
  }
  Strides concat(Strides other) const {
    Strides res(ndims + other.ndims);
    for (int i=0;i<ndims;i++) res[i] = num_bytes[i];
    for (int i=0;i<other.ndims;i++) res[i+ndims] = other.num_bytes[i];
    return res;
  }
  Strides reverse() const {
    Strides res = *this;
    std::reverse(std::begin(res.num_bytes), std::begin(res.num_bytes)+ndims);
    return res;
  }
  bool operator==(const Strides& other) const {
    if (ndims != other.ndims) return false;
    for (int i=0;i<ndims;i++)
      if (num_bytes[i] != other.num_bytes[i])
        return false;
    return true;
  }
  bool operator!=(const Strides& other) const {
    return !(*this == other);
  }
  bool is_contiguous(DataType datatype) const {
    return *this == Strides(datatype.get_shape(), datatype.get_element_type());
  }
  bool is_reverse_contiguous(DataType datatype) const {
    return *this == Strides(datatype.get_shape(), datatype.get_element_type()).reverse();
  }
private:
  size_t ndims{};
  Array num_bytes{};
};

inline std::ostream& operator<<(std::ostream& os, Strides strides) {
  os << "[";
  for (int i=0;i<strides.get_ndims();i++){
    if (i != strides.get_ndims()-1)
      os << strides[i] << ", ";
    else
      os << strides[i];
  }
  return os << "]";
}

inline std::ostream& operator<<(std::ostream& os, DataType datatype) {
  return os << datatype.get_element_type() << datatype.get_shape();
}

class TensorDescriptor {
public:
  TensorDescriptor() = default;
  TensorDescriptor(ElementType element_type, Shape shape, Strides strides) :
    element_type(element_type), shape(shape), strides(strides) {}
  TensorDescriptor(Shape shape, ElementType element_type) : 
    element_type(element_type), shape(shape), strides(shape, element_type) {}
  DataType get_datatype() const {
    return DataType(element_type, shape);
  }
  size_t get_data_size() const {
    return shape[0] * strides[0]; 
  }
  size_t get_ndims() const { return shape.get_ndims(); }
  ElementType get_element_type() const { return element_type; }
  Shape get_shape() const { return shape; }
  Strides get_strides() const { return strides; }
  TensorDescriptor set_shape(Shape shape) const {
    TensorDescriptor res = *this;
    res.shape = shape;
    res.strides = Strides(shape, element_type);
    return res;
  }
private:
  ElementType element_type{};
  Shape shape{};
  Strides strides{};
};

template <typename T>
class type_wrapper {
public:
};

template<typename Ret>
static inline Ret call_with_type(auto&& func, ElementType element_type) {
  switch (element_type) {
  #define ELEMENT_TYPE(tyty, name, enum_name, bytes_size) \
  case ElementType::enum_name: \
  return func(type_wrapper<tyty>());\
  break;
  #include "element_type.inc"
  #undef ELEMENT_TYPE
  case ElementType::None:
    ASSERT(false, "none element type used");
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
  bool operator==(const Constant& other) const {
    return type == other.type && val == other.val;
  }
  bool operator!=(const Constant& other) const {
    return !(Constant::operator==(other));
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

}
}