#pragma once

#include <stack>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

namespace katoml {
namespace compiler {

static inline std::string pretty_indent(const std::string& str, int max_width = 64) {
  int root = -1;
  std::unordered_map<int, std::vector<int>> g;
  std::unordered_map<int, int> open_to_close;
  std::stack<int> st;
  const std::string TAB = "  ";
  for (int i=0;i<str.size();i++){
    if (str[i] == '(' || str[i] == '[') {
      if (!st.empty()) {
        g[st.top()].push_back(i);
      }
      st.push(i);
      if (root == -1) {
        root = i;
      }
    } else if (str[i] == ')' || str[i] == ']') {
      open_to_close[st.top()] = i;
      st.pop();
    }
  }
  const auto dfs = [&](auto&& self, int cur, int start, int end) -> std::vector<std::string> {
    if (end - start + 1 > max_width) {
      std::vector<std::string> res;
      res.push_back(str.substr(start,cur-start+1));
      int last = cur+1;
      for (int v : g[cur]) {
        auto child = self(self, v, last, open_to_close[v]);
        for (std::string s : child) {
          res.push_back(TAB + s);
        }
        last = open_to_close[v]+1;
        if (str[last] == ',') {
          res.back().push_back(',');
          last++;
        }
        while (last < end && str[last] == ' ') {
          last++;
        }
      }
      if (last != end)
        res.push_back(TAB + str.substr(last, end-last));
      res.push_back(std::string(1,str[end]));
      return res;
    } else {
      return std::vector<std::string>({str.substr(start, end-start+1)});
    }
  };
  if (root != -1) {
    std::ostringstream ss("");
    auto res = dfs(dfs, root, 0, str.size()-1);
    for (auto s : res) {
      ss << s << "\n";
    }
    return ss.str();
  }
  return str;
}

template<typename T>
static inline std::string to_string(const T& value) {
  std::ostringstream ss("");
  ss << value;
  return ss.str();
}

}
}