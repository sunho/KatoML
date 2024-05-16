#include <pybind11/pybind11.h>

#include <katoml/mlcompiler/mlcompiler.hpp>

using namespace katoml;
using namespace tensor;

namespace py = pybind11;

tensor::Backend& backend = compiler::default_device->backend();

static Tensor create_tensor(Shape shape, ElementType element_type) {
  return backend.zeros(DataType(element_type, shape));
}

std::vector<int64_t> tuple_to_vector(const py::tuple& tuple) {
    std::vector<int64_t> vec(tuple.size());
    for (size_t i = 0; i < tuple.size(); ++i) {
        vec[i] = tuple[i].cast<int64_t>();
    }
    return vec;
}

std::vector<int> tuple_to_ivector(const py::tuple& tuple) {
    std::vector<int> vec(tuple.size());
    for (size_t i = 0; i < tuple.size(); ++i) {
        vec[i] = tuple[i].cast<int>();
    }
    return vec;
}

Constant handle_indexing(const Tensor& t, const py::tuple& indices) {
  // Determine if we are working with slices, integers, or a combination
  /* bool is_slice = false; */
  /* for (size_t i = 0; i < indices.size(); ++i) { */
  /*     if (py::isinstance<py::slice>(indices[i])) { */
  /*         is_slice = true; */
  /*         break; */
  /*     } */
  /* } */
  /**/
  /* if (is_slice) { */
  /*     // Handle slicing */
  /*     std::vector<py::slice> slices = to_slice_vector(indices); */
  /*     Shape shape = t.get_shape(); */
  /*     std::vector<int64_t> new_shape; */
  /**/
  /*     for (size_t i = 0; i < shape.get_ndims(); ++i) { */
  /*         if (i >= slices.size()) break; */
  /*         auto slice = slices[i]; */
  /*         auto indices = slice.indices(shape[i]); */
  /*         auto start = indices[0]; */
  /*         auto stop = indices[1]; */
  /*         auto step = indices[2]; */
  /**/
  /*         int64_t dim_size = (stop - start + step - 1) / step; */
  /*         new_shape.push_back(dim_size); */
  /*     } */
  /**/
  /*     // Build a new tensor with the sliced data */
  /*     Shape new_tensor_shape(new_shape.size(), Shape::Array{new_shape.begin(), new_shape.end()}); */
  /*     auto new_tensor = Tensor(t.backend.get(), t.backend.get().zeros(Resource::DataType(t.get_element_type(), new_tensor_shape))); */
  /**/
  /*     // Copy data into the new tensor (implement your specific copy logic here) */
  /*     // ... */
  /**/
  /*     return new_tensor; */
  /* } else { */
  /*     // Handle multi-dimensional indices */
  const auto index_vec = tuple_to_ivector(indices);
  return t.at_slow(index_vec);
  /* } */
}

void set_at_indices(Tensor& t, const py::tuple& indices, Constant c) {
  const auto index_vec = tuple_to_ivector(indices);
  t.at_slow(index_vec) = c;
}

namespace pybind11::detail {
  template <> struct type_caster<Constant> : public type_caster_base<Constant> {
    using base = type_caster_base<Constant>;
  public:
    bool load(handle src, bool) {
      if (py::isinstance<py::int_>(src)) {
        value = new Constant(py::cast<int>(src));
        return true;
      } else if (py::isinstance<py::float_>(src)) {
          value = new Constant(py::cast<float>(src));
          return true;
      } else {
        return false;
      }
    }
    static handle cast(const Constant& src, return_value_policy policy, handle parent) {
      switch (src.get_type()) {
      case ElementType::Float32:
      case ElementType::Float64:
        return py::cast(src.cast<double>(), policy, parent).release();
      case ElementType::Int32:
      case ElementType::Int64:
        return py::cast(src.cast<int64_t>(), policy, parent).release();
      case ElementType::None:
        return nullptr;
      }
    }
  };

  template <> struct type_caster<Shape> : public type_caster_base<Shape> {
    using base = type_caster_base<Shape>;
  public:
    bool load(handle src, bool convert) {
      if (base::load(src, convert)) {
        return true;
      }
      if (py::isinstance<py::int_>(src)) {
        value = new Shape({py::cast<int>(src)});
        return true;
      } else if (py::isinstance<py::tuple>(src)) {
        auto vec = tuple_to_vector(py::cast<py::tuple>(src));
        Shape::Array arr{};
        std::copy(vec.begin(), vec.end(), arr.begin());
        value = new Shape(vec.size(), arr);
        return true;
      } else {
        return false;
      }
    }
    static handle cast(const Shape& src, return_value_policy policy, handle parent) {
      return base::cast(src, policy, parent);
    }
  };
}

#define BIND_NATIVE_OP_COPY(pyname, opr) .def(pyname, [](Tensor &t, Constant c) -> Tensor {\
  return t.operator opr(c);\
}).def(pyname, [](Tensor &t, const Tensor& c) -> Tensor {\
  return t.operator opr(c);\
})

#define BIND_NATIVE_OP(pyname, opr) .def(pyname, [](Tensor &t, Constant c) -> Tensor& {\
  return t.operator opr(c);\
}).def(pyname, [](Tensor &t, const Tensor& c) -> Tensor& {\
  return t.operator opr(c);\
})

PYBIND11_MODULE(katoml, m) {
  py::class_<Constant>(m, "Constant");

  py::enum_<ElementType>(m, "ElementType")
    .value("float32", ElementType::Float32)
    .value("float64", ElementType::Float64)
    .value("int32", ElementType::Int32)
    .value("int64", ElementType::Int64);

  py::class_<Strides>(m, "Strides")
    .def("get_ndims", &Strides::get_ndims)
    .def("get_num_bytes", &Strides::get_num_bytes)
    .def("__getitem__", [](const Strides& s, int i) {
        return s[i];
    })
    .def("__setitem__", [](Strides& s, int i, size_t val) {
        s[i] = val;
    })
    .def("__repr__", [](const Strides& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("__str__", [](const Strides& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("slice", &Strides::slice)
    .def("concat", &Strides::concat)
    .def("reverse", &Strides::reverse)
    .def("__eq__", &Strides::operator==)
    .def("__ne__", &Strides::operator!=)
    .def("is_contiguous", &Strides::is_contiguous)
    .def("is_reverse_contiguous", &Strides::is_reverse_contiguous);

   py::class_<Shape>(m, "Shape")
    .def("get_ndims", &Shape::get_ndims)
    .def("get_num_elements", &Shape::get_num_elements)
    .def("get_total", &Shape::get_total)
    .def("get_total_without_any", &Shape::get_total_without_any)
    .def("__repr__", [](const Shape& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("__str__", [](const Shape& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("__getitem__", [](const Shape& s, int i) {
        return s[i];
    })
    .def("__setitem__", [](Shape& s, int i, int64_t val) {
        s[i] = val;
    })
    .def("slice", &Shape::slice)
    .def("concat", &Shape::concat)
    .def("insert_axis", &Shape::insert_axis)
    .def("extend_axis", &Shape::extend_axis)
    .def("reverse", &Shape::reverse)
    .def("reduce", &Shape::reduce)
    .def("normalize_axis", &Shape::normalize_axis)
    .def("compatible", &Shape::compatible)
    .def("__eq__", &Shape::operator==)
    .def("__ne__", &Shape::operator!=)
    .def_property("Any", [](py::object &) { return Shape::Any; }, nullptr);

  py::class_<katoml::tensor::Tensor>(m, "Tensor")
    .def(py::init(&create_tensor),
        py::arg("shape")=Shape({1}), py::arg("element_type")=ElementType::Float32
    )
    .def("__repr__", [](const Tensor& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("__str__", [](const Tensor& s) {
      std::stringstream ss;
      ss << s;
      return ss.str();
    })
    .def("get_ndims", &tensor::Tensor::get_ndims)
    .def("get_element_type", &tensor::Tensor::get_element_type)
    .def("get_shape", &tensor::Tensor::get_shape)
    .def("get_strides", &tensor::Tensor::get_strides)
    .def("__getitem__", [](const Tensor &t, py::tuple indices) {
        return handle_indexing(t, indices);
    })
    .def("__getitem__", [](const Tensor &t, int i) {
        return t.at(i);
    })
    .def("__setitem__", [](Tensor &t, py::tuple indices, Constant c) {
        set_at_indices(t, indices, c);
    })
    .def("__setitem__", [](Tensor &t, int i, Constant c) {
        t.at(i) = c;
    })
    .def("clip", &katoml::tensor::Tensor::clip)
    .def("reshaped", &katoml::tensor::Tensor::reshaped)
    .def("transposed", &katoml::tensor::Tensor::transposed)
    .def("axis_extended", &katoml::tensor::Tensor::axis_extended)
    .def("reshape", &katoml::tensor::Tensor::reshape)
    .def("transpose", &katoml::tensor::Tensor::transpose)
    .def("extend_axis", &katoml::tensor::Tensor::extend_axis)
    .def("near_equals", &katoml::tensor::Tensor::near_equals)
    .def("matmul", &katoml::tensor::Tensor::matmul)
    BIND_NATIVE_OP_COPY("__add__", +)
    BIND_NATIVE_OP_COPY("__sub__", -)
    BIND_NATIVE_OP_COPY("__mul__", *)
    BIND_NATIVE_OP_COPY("__truediv__", /)
    BIND_NATIVE_OP_COPY("__lt__", <)
    BIND_NATIVE_OP_COPY("__le__", <=)
    BIND_NATIVE_OP_COPY("__gt__", >)
    BIND_NATIVE_OP_COPY("__ge__", >=)
    BIND_NATIVE_OP_COPY("__eq__", ==)
    BIND_NATIVE_OP("__iadd__", +=)
    BIND_NATIVE_OP("__isub__", -=)
    BIND_NATIVE_OP("__imul__", *=)
    BIND_NATIVE_OP("__itruediv__", /=)
    .def("sum", &katoml::tensor::Tensor::sum)
    .def("mean", &katoml::tensor::Tensor::mean)
    .def("reduce_max", &katoml::tensor::Tensor::reduce_max)
    .def("reduce_min", &katoml::tensor::Tensor::reduce_min)
    .def("__neg__", [](const Tensor &t) -> Tensor {\
      return -t;
    })
    .def("log", &katoml::tensor::Tensor::log)
    .def("exp", &katoml::tensor::Tensor::exp)
    .def("diag", &katoml::tensor::Tensor::diag)
    .def("copy", &katoml::tensor::Tensor::copy);
}


