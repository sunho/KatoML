#pragma once

#include "types.hpp"

namespace katoml {
namespace tensor {

class IterUtils {
public:
  struct WOffsetView {
    WOffsetView(void* data, TensorDescriptor descriptor) : 
      data(data), 
      shape(descriptor.get_shape()), 
      strides(descriptor.get_strides()) {}
    WOffsetView(void* data, size_t offset, Shape shape, Strides strides) : 
      data(data),
      offset(offset),
      shape(shape),
      strides(strides) {}
    template<typename T>
    T read(size_t addend = 0) const { 
      return *reinterpret_cast<const T*>((const uint8_t*)data + offset + addend);
    }
    template<typename T>
    void write(T a, size_t addend = 0) {
      *reinterpret_cast<T*>((uint8_t*)data + offset + addend) = a;
    }
    size_t offset{};
    void* data{};
    Shape shape{};
    Strides strides{};
  };

  struct ROffsetView {
    ROffsetView(const void* data, TensorDescriptor descriptor) : 
      data(data),
      shape(descriptor.get_shape()), 
      strides(descriptor.get_strides()) {}
    ROffsetView(const void* data, size_t offset, Shape shape, Strides strides) : 
      data(data),
      offset(offset),
      shape(shape),
      strides(strides) {}
    template<typename T>
    T read(size_t addend = 0) const { 
      return *reinterpret_cast<const T*>((const uint8_t*)data + offset + addend);
    }
    size_t offset{};
    const void* data{};
    Shape shape{};
    Strides strides{};
  };

  template<class T, auto operation>
  static void per_element_bin_op(WOffsetView res, ROffsetView lhs, ROffsetView rhs) {
    per_element_iterate_bin_op<T,operation>(0, res, lhs, rhs);
  }

  template<class T, auto operation>
  static void per_element_iterate_bin_op(int level, WOffsetView res, ROffsetView lhs, ROffsetView rhs) {
    if (level == res.shape.get_ndims()) {
      res.write(operation(lhs.read<T>(), rhs.read<T>()));
      return;
    }
    for (int64_t i=0;i<res.shape[level];i++) {
      per_element_iterate_bin_op<T,operation>(level+1, res, lhs, rhs);
      res.offset += res.strides[level];
      lhs.offset += lhs.strides[level];
      rhs.offset += rhs.strides[level];
    }
  }

  template<class T, auto operation>
  static bool all(ROffsetView lhs, ROffsetView rhs) {
    return all_iterate<T,operation>(0, lhs, rhs);
  }

  template<class T, auto operation>
  static bool all_iterate(int level, ROffsetView lhs, ROffsetView rhs) {
    if (level == lhs.shape.get_ndims()) {
      if (!operation(lhs.read<T>(), rhs.read<T>())) 
        return false;
      return true;
    }
    for (int64_t i=0;i<lhs.shape[level];i++) {
      if (!all_iterate<T,operation>(level+1, lhs, rhs))
        return false;
      lhs.offset += lhs.strides[level];
      rhs.offset += rhs.strides[level];
    }
    return true;
  }

  template<class T, auto operation>
  static void per_element_uni_op(WOffsetView res, ROffsetView val) {
    const auto override = [](T a, int){ return operation(a); };
    per_element_iterate_uni_op<T,int,override>(0, res, val,0);
  }

  template<class T, class UserData, auto operation>
  static void per_element_uni_op(WOffsetView res, ROffsetView val, UserData user_data) {
    per_element_iterate_uni_op<T,UserData,operation>(0, res, val, user_data);
  }

  template<class T, class UserData, auto operation>
  static void per_element_iterate_uni_op(int level, WOffsetView res, ROffsetView val, UserData user_data) {
    if (level == res.shape.get_ndims()) {
      res.write(operation(val.read<T>(), user_data));
      return;
    }
    for (size_t i=0;i<res.shape[level];i++) {
      per_element_iterate_uni_op<T,UserData,operation>(level+1, res, val, user_data);
      res.offset += res.strides[level];
      val.offset += val.strides[level];
    }
  }

  template<class T, auto operation>
  static void per_diag_element_uni_op(WOffsetView res, ROffsetView val) {
    for (size_t i=0;i<val.shape[0];i++) {
      res.write<T>(operation(res.read<T>()));
      res.offset += res.strides[0];
      res.offset += res.strides[1];
      val.offset += val.strides[0];
    }
  }

  template<class T, auto operation>
  static void per_element_self(WOffsetView res) {
    const auto override = [](T a, int){ return operation(a); };
    per_element_iterate_self<T,int,override>(0, res, 0);
  }

  template<class T, class UserData, auto operation>
  static void per_element_self(WOffsetView res, UserData user_data) {
    per_element_iterate_self<T,UserData,operation>(0, res, user_data);
  }

  template<class T, class UserData, auto operation>
  static void per_element_iterate_self(int level, WOffsetView res, UserData user_data) {
    if (level == res.shape.get_ndims()) {
      res.write(operation(res.read<T>(), user_data));
      return;
    }
    for (size_t i=0;i<res.shape[level];i++) {
      per_element_iterate_self<T,UserData,operation>(level+1, res, user_data);
      res.offset += res.strides[level];
    }
  }

  template<class T, auto operation>
  static void reduce(WOffsetView res, ROffsetView val, const std::vector<int>& axis) {
    reduce_iterate<T,operation>(0, res, val);
  }

  template<class T, auto operation>
  static void reduce_iterate(int level, WOffsetView res, ROffsetView val) {
    if (level == val.shape.get_ndims()) {
      res.write(operation(res.read<T>(), val.read<T>()));
      return;
    }
    for (size_t i=0;i<val.shape[level];i++) {
      reduce_iterate<T,operation>(level+1, res, val);
      res.offset += res.strides[level];
      val.offset += val.strides[level];
    }
  }

  template<class T, auto operation>
  static void reduce_self(WOffsetView res, ROffsetView val) {
    const auto override = [](T a, int){ return operation(a); };
    reduce_self<T,int,override>(res, val, 0);
  }

  template<class T, class UserData, auto operation>
  static void reduce_self(WOffsetView res, ROffsetView val, UserData user_data) {
    reduce_self_iterate<T,UserData,operation>(0, res, val, user_data);
  }

  template<class T, class UserData, auto operation>
  static void reduce_self_iterate(int level, WOffsetView res, ROffsetView val, UserData user_data) {
    if (level == val.shape.get_ndims()) {
      res.write(operation(res.read<T>(), user_data));
      return;
    }
    for (size_t i=0;i<val.shape[level];i++) {
      reduce_self_iterate<T,UserData,operation>(level+1, res, val, user_data);
      res.offset += res.strides[level];
      val.offset += val.strides[level];
    }
  }

  using PerElementLevelInFunc = std::function<void(int level)>;
  using PerElementLevelOutFunc = std::function<void(int level,bool last)>;
  template<class T>
  using PerElementReadFunc = std::function<void(T val,bool last)>;
  template<class T>
  static void per_element_read(ROffsetView val, PerElementLevelInFunc level_in, PerElementLevelOutFunc level_out, PerElementReadFunc<T> read) {
    level_in(0);
    per_element_read_iterate<T>(0, val, level_in, level_out, read);
    level_out(0, true);
  }

  template<class T>
  static void per_element_read_iterate(int level, ROffsetView val, PerElementLevelInFunc level_in, PerElementLevelOutFunc level_out, PerElementReadFunc<T> read) {
    for (int64_t i=0;i<val.shape[level];i++) {
      if (level + 1 == val.shape.get_ndims()) {
        read(val.read<T>(), i == val.shape[level] - 1);
      } else {
        level_in(level+1);
        per_element_read_iterate<T>(level+1, val, level_in, level_out, read);
        level_out(level+1, i == val.shape[level] - 1);
      }
      val.offset += val.strides[level];
    }
  }

  template<class T>
  static void matmul(IterUtils::WOffsetView res, IterUtils::ROffsetView lhs, IterUtils::ROffsetView rhs) {
    matmul_iterate<T>(0, res, lhs, rhs);
  }

  template<class T>
  static void matmul_iterate(int level, IterUtils::WOffsetView res, IterUtils::ROffsetView lhs, IterUtils::ROffsetView rhs) {
    if (level == res.shape.get_ndims()-2) {
      int64_t n = lhs.shape[-2];
      int64_t p = lhs.shape[-1];
      int64_t m = rhs.shape[-1];
      for (int64_t k = 0, l_offset_out = 0, r_offset_out = 0; k < p; 
          k++, l_offset_out+=lhs.strides[level+1], r_offset_out+=rhs.strides[level]) {
        for (int64_t i = 0, l_offset = l_offset_out, r_offset = r_offset_out, res_offset = 0; i < n; 
            i++, res_offset += res.strides[level], l_offset += lhs.strides[level]) {
          for (int64_t j = 0, l_offset_=l_offset, r_offset_ = r_offset, res_offset_ = res_offset; j < m; 
              j++, res_offset_ += res.strides[level+1], r_offset_ += rhs.strides[level+1]) {
            res.write(res.read<T>(res_offset_) + lhs.read<T>(l_offset_) * rhs.read<T>(r_offset_), res_offset_);
          }
        }
      }
      return;
    } 
    for (int64_t i=0;i<res.shape[level];i++) {
      matmul_iterate<T>(level+1, res, lhs, rhs);
      res.offset += res.strides[level];
      lhs.offset += lhs.strides[level];
      rhs.offset += rhs.strides[level];
    }
  }

};

}
}