#pragma once
#include "value.hpp"
#include "node.hpp"

namespace katoml {
namespace compiler {
namespace ir {

template <typename Ret, typename T, size_t N, typename... Args, size_t... I>
Ret callVisitorImpl(T &obj, Ret (T::*mf)(Args...), const std::array<Value, N> &args,
                     std::index_sequence<I...>) {
  return (obj.*mf)(args[I]...);
}

template <typename Ret, typename T, auto mf, size_t N>
Ret callVisitor(T &obj, const std::array<Value, N> &args) {
  return callVisitorImpl(obj, mf, args, std::make_index_sequence<N>{});
}

template <typename... Args>
struct SizeGetter {
  constexpr static size_t size = sizeof...(Args);
};

template <typename Ret, typename Impl>
struct NodeVisitorCaller {
  Ret call(Impl &impl, const Node &node) {
      switch (node.get_opcode()) {
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) \
  case Opcode::OP: \
      return _call##OP(impl, node); \
      break;
#include "ir.inc"
#undef DECL_NODE
    }
  }

#define DECL_NODE(OP, ARGS, PARAMS, TYPES)  \
  Ret _call##OP(Impl &impl, const Node& node) {  \
    std::array<Value, SizeGetter<TYPES>::size> tmp; \
    for (int i = 0; i < SizeGetter<TYPES>::size; ++i) { \
      tmp[i] = node.get_args()[i];  \
    }  \
    return callVisitor<Ret, Impl, &Impl::OP, SizeGetter<TYPES>::size>(impl, tmp);  \
  }
#include "ir.inc"
#undef DECL_NODE
};

template <typename Ret, typename UserData, typename T, size_t N, typename... Args, size_t... I>
Ret callVisitorImpl(T &obj, Ret (T::*mf)(Args...), const std::array<Value, N> &args, UserData& user_data,
                     std::index_sequence<I...>) {
  return (obj.*mf)(user_data, args[I]...);
}

template <typename Ret, typename UserData, typename T, auto mf, size_t N>
Ret callVisitor(T &obj, const std::array<Value, N> &args, UserData& user_data) {
  return callVisitorImpl(obj, mf, args, user_data, std::make_index_sequence<N>{});
}

template <typename Ret, typename UserData, typename Impl>
struct NodeVisitorCallerWithUserData {
  Ret call(Impl &impl, const Node &node, UserData& user_data) {
      switch (node.get_opcode()) {
#define DECL_NODE(OP, ARGS, PARAMS, TYPES) \
  case Opcode::OP: \
      return _call##OP(impl, node, user_data); \
      break;
#include "ir.inc"
#undef DECL_NODE
    }
  }

#define DECL_NODE(OP, ARGS, PARAMS, TYPES)  \
  Ret _call##OP(Impl &impl, const Node& node, UserData& user_data) {  \
    std::array<Value, SizeGetter<TYPES>::size> tmp; \
    for (int i = 0; i < SizeGetter<TYPES>::size; ++i) { \
      tmp[i] = node.get_args()[i];  \
    }  \
    return callVisitor<Ret, UserData, Impl, &Impl::OP, SizeGetter<TYPES>::size>(impl, tmp, user_data);  \
  }
#include "ir.inc"
#undef DECL_NODE
};


}
}
}