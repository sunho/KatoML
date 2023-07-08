#pragma once

#include <stdexcept>
#include <iostream>
#include <sstream>

namespace katoml {

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
ERROR_DEF(UseAfterFreeError, "tried to use view to already released tensor")
ERROR_DEF(ViewAssignAllocationError, "modify-assign operation on view needed new allocation")
ERROR_DEF(NullConstantError, "tried to cast null tensor constant")
ERROR_DEF(InvalidTypedError, "tried to assign mistyped value to Typed container")
ERROR_DEF(ExecutorInternalError, "executor internal error")
ERROR_DEF(InvalidReduceAxisError, "invalid reduce axis")

ERROR_DEF(NotEnoughNNInputsError, "not enough inputs to neural network")
ERROR_DEF(WrongNNInputsError, "wrong input to neural network")
ERROR_DEF(NonSingleNNOutputError, "output of neural network model must be exactly one")
ERROR_DEF(DuplicateNNInputError, "duplicate inputs with same name defined inside neural network model")
ERROR_DEF(UninitNetworkContextError, "thread network context not initialized")
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