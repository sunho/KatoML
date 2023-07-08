#include <katoml/mlcompiler/device.hpp>

using namespace katoml;
using namespace katoml::compiler;
using namespace katoml::tensor;

int main() {
  auto device = construct_device();
  
  auto A = device->constant([&](){
    auto A = device->backend().zeros<float>(3,3);
    for (int i=0;i<3;i++)
      A(i,i) = 3.0;
    return A;
  }());
  auto A2 = device->matmul(A,A) + A;

  std::cout << "Intermediate Representation: " << "\n";
  std::cout << pretty_indent(to_string(A2)) << "\n";

  auto program = device->compile(A2);
  std::cout << "Evaluated: " << "\n";
  std::cout << program->forward() << "\n";
}