#pragma once

#include <stdexcept>
#include <iostream>
#include <sstream>

namespace katoml {
namespace tensor {

class RuntimeError : public std::runtime_error {
protected:
  RuntimeError(const std::string& msg) : std::runtime_error(msg) {}
  RuntimeError(const char* msg) : std::runtime_error(msg) {}
};

#define ERROR_DEF(class_name, msg) \
class class_name final : public RuntimeError {\
public:\
  class_name() : RuntimeError(msg) {}\
};

ERROR_DEF(BackendMismatchError, "tried to operate with tensors with mistmatching backends")
ERROR_DEF(NullConstantError, "tried to cast null tensor constant")
ERROR_DEF(TensorTypeError, "tried to assign mistyped tensor to TypedTensor")
ERROR_DEF(ExecutorInternalError, "executor internal error")
ERROR_DEF(InvalidReduceAxisError, "invalid reduce axis")
#undef ERROR_DEF

class FatalRuntimeError final : public RuntimeError {
public:
  FatalRuntimeError(const std::string& msg) : RuntimeError(msg) {}
};

class DataTypeMisMatchError final : public RuntimeError {
public:
  template<typename... DT>
  DataTypeMisMatchError(const std::string& operation, DT... dt) : RuntimeError(create_error_string(operation, dt...)) {}

private:
  template<typename... DT>
  static inline std::string create_error_string(const std::string& operation, DT... dt) {
    std::stringstream os("");
    os << "not compatible datatypes for " << operation << ": ";
    ((os << dt,os<<"  "),...);
    return os.str();
  }
};

static inline void panic(const std::string& msg) {
  std::cerr << "FATAL ERROR: " << msg << std::endl;
  throw FatalRuntimeError(msg);
}

#define ASSERT(eval, msg) assert((eval) && msg);
#define ASSERT_FAIL(eval, msg) { if (!(eval)) panic(msg); }
#define CHECK_OR_THROW(eval, err) { if (!(eval)) throw err(); }
#define TYPE_CHECK_OR_THROW(eval, operation, ...) { if (!(eval)) throw DataTypeMisMatchError((operation), __VA_ARGS__); }

}
}