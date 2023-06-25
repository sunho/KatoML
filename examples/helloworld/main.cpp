#include <katoml/mlcompiler/device.hpp>
#include <katoml/mlcompiler/utils/string_utils.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;

int main() {
  auto device = construct_device();
  auto& graph = device->graph();
  
  auto A = graph.constant([&](){
    auto A = device->zeros<float>(3,3);
    for (int i=0;i<3;i++)
      A(i,i) = 3.0;
    return A;
  }());
  auto A2 = graph.matmul(A,A) + A;

  std::cout << "Intermediate Representation: " << "\n";
  std::cout << pretty_indent(to_string(A2)) << "\n";

  auto program = device->compile(A2);
  std::cout << "Evaluated: " << "\n";
  std::cout << program.forward() << "\n";
}