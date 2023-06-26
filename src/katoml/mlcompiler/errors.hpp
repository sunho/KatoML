#pragma once
#include <stdexcept>

namespace katoml {
namespace compiler {

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
#undef ERROR_DEF

}
}