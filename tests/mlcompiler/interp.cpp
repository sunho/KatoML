#include "../common.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mlcompiler/mlcompiler.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;

static auto device = construct_device();

#define evaluate(eval) (device->compile(eval)->forward())

TEST_CASE("[mlcompiler] Add is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A + A), EqualsTensor(device->backend().tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A + B), EqualsTensor(device->backend().tensor<int>({{2,3,4},{2,3,4},{2,3,4}})));

  auto C = device->tensor<int>({1});
  REQUIRE_THAT(evaluate(A + C), EqualsTensor(device->backend().tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  auto row = device->tensor<int>({1,2,3});
  auto col = device->tensor<int>({{10},{20},{30}});
  REQUIRE_THAT(evaluate(row + col), EqualsTensor(device->backend().tensor<int>({{11,12,13},{21,22,23},{31,32,33}})));
}

TEST_CASE("[mlcompiler] Sub is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A - A), EqualsTensor(device->backend().tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A - B), EqualsTensor(device->backend().tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}})));
}

TEST_CASE("[mlcompiler] Mul is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A * A), EqualsTensor(device->backend().tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A * B), EqualsTensor(device->backend().tensor<int>({{1,2,3},{1,2,3},{1,2,3}})));
}

TEST_CASE("[mlcompiler] Div is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A / A), EqualsTensor(device->backend().tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A / B), EqualsTensor(device->backend().tensor<int>({{1,0,0},{1,0,0},{1,0,0}})));
}

TEST_CASE("[mlcompiler] Softmax is computed") {
  auto A = device->tensor<float>({1,2,3});
  REQUIRE_THAT(evaluate(A.softmax()), ApproxTensor(device->backend().tensor<float>({0.09003057, 0.24472847, 0.66524096})));

  auto B =  device->tensor<float>({{1, 0.5, 0.2, 3},{1,  -1,   7, 3},{2,  12,  13, 3}});
  REQUIRE_THAT(evaluate(B.softmax()), ApproxTensor(device->backend().tensor<float>({{1.05877070e-01,6.42176889e-02,4.75736340e-02,7.82331607e-01},{2.42746030e-03,3.28521027e-04,9.79307378e-01,1.79366403e-02},{1.22093673e-05,2.68929212e-01,7.31025390e-01,3.31885014e-05}})));
}

TEST_CASE("[mlcompiler] Log Softmax backwards") {
  auto W = device->var(device->backend().tensor<float>({1,2,3}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE_THAT(program->forward(), ApproxTensor(device->backend().tensor<float>({-2.40760596,-1.40760596,-0.40760596})));
  program->backward();
  REQUIRE_THAT(W.get_grad(), ApproxTensor(device->backend().tensor<float>({-7.15484549,-21.1671683,-59.25661077})));
}

TEST_CASE("[mlcompiler] Log Softmax backwards 2") {
  auto W = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6}}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE_THAT(program->forward(), ApproxTensor(device->backend().tensor<float>({{-2.40760596,-1.40760596,-0.40760596},{-2.40760596,-1.40760596,-0.40760596}})));
  program->backward();
  REQUIRE_THAT(W.get_grad(), ApproxTensor(device->backend().tensor<float>({{-7.15484549,-21.1671683,-59.25661077},{-162.7944501,-444.23947731,-1209.28638048}})));
}

TEST_CASE("[mlcompiler] Mat mul backwards") {
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto B = device->var(device->backend().tensor<float>({{3,2,1},{6,5,4},{7,8,9}}));
  auto matmul = device->matmul(A, B);
  auto program = device->compile(matmul);
  REQUIRE_THAT(program->forward(), ApproxTensor(device->backend().tensor<float>({{36,36 ,36},{84,81,78},{132,126,120}})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), EqualsTensor(device->backend().tensor<float>({{6,15,24},{6,15,24},{6,15,24}})));
  REQUIRE_THAT(B.get_grad(), EqualsTensor(device->backend().tensor<float>({{12,12,12},{15,15,15},{18,18,18}})));
}

TEST_CASE("[mlcompiler] Opertaions with constant is computed") {
  auto A = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(10 * A), EqualsTensor(device->backend().tensor<int>({10,20,30})));
  REQUIRE_THAT(evaluate(A * 10), EqualsTensor(device->backend().tensor<int>({10,20,30})));
  REQUIRE_THAT(evaluate(A + 10), EqualsTensor(device->backend().tensor<int>({11,12,13})));
  REQUIRE_THAT(evaluate(A.max(2)), EqualsTensor(device->backend().tensor<int>({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(A, 2)), EqualsTensor(device->backend().tensor<int>({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(2, A)), EqualsTensor(device->backend().tensor<int>({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(A, A)), EqualsTensor(device->backend().tensor<int>({1,2,3})));
}

TEST_CASE("[mlcompiler] Log softmax sum backwards") {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto sum1 = device->sum(A, {0});
  auto sf1 = device->log_softmax(sum1);
  auto sum2 = device->sum(A, {1});
  auto sf2 = device->log_softmax(sum2);
  auto program = device->compile(sf1);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(device->backend().tensor<float>({{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08},{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08},{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08}})));
  program = device->compile(sf2);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(device->backend().tensor<float>({{-1.20928638e+03,-1.20928638e+03,-1.20928638e+03},{-9.80705112e+06,-9.80705112e+06,-9.80705112e+06},{-7.94673664e+10,-7.94673664e+10,-7.94673664e+10}})));
}

TEST_CASE("[mlcompiler] Log softmax mean backwards") {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto mean1 = device->mean(A, {0});
  auto sf1 = device->log_softmax(mean1);
  auto mean2 = device->mean(A, {1});
  auto sf2 = device->log_softmax(mean2);
  auto program = device->compile(sf1);
  REQUIRE_THAT(program->forward(), ApproxTensor(device->backend().tensor<float>({-2.40760596, -1.40760596, -0.40760596})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(device->backend().tensor<float>({{-54.2648167,-148.07982577,-403.09546016},{-54.2648167,-148.07982577,-403.09546016},{-54.2648167,-148.07982577,-403.09546016}})));
  program = device->compile(sf2);
  REQUIRE_THAT(program->forward(), ApproxTensor(device->backend().tensor<float>({-6.05094576,-3.05094576,-0.0509457})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(device->backend().tensor<float>({{-7.05572277,-7.05572277,-7.05572277},{-148.07982577,-148.07982577,-148.07982577},{-2980.62465371,-2980.62465371,-2980.62465371}})));
}
