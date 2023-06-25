#pragma once

#include <stack>
#include <string>
#include <stdexcept>
#include <sstream>

namespace katoml {
namespace compiler {

template<typename... Args>
std::string format(const std::string& format_text, Args...args) {
  std::ostringstream os("");
  int cur = 0;
  auto feed = [&]() {
    bool done = false;
    while (cur < format_text.size()) {
      if (format_text[cur] == '{' && cur + 1 < format_text.size() && format_text[cur + 1] == '}') {
        cur += 2;
        done = true;
        break;
      }
      os << format_text[cur];
      cur ++;
    }
    return done;
  };
  auto feed_or_throw = [&]() {
    if (!feed()) 
      throw std::runtime_error("format text has less {} than given arguments");
  };
  ((feed_or_throw(),os << args),...);
  if (feed())
    throw std::runtime_error("format text has more {} than given arguments");
  return os.str();
}

}
}