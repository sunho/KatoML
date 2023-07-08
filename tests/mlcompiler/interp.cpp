#include "katoml/mltensor/core.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mlcompiler/mlcompiler.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;

static auto device = construct_device();

#define evaluate(eval) (device->compile(eval)->forward())

TEST_CASE( "Add is computed", "[cpu]" ) {
  auto A = device->ones_i32(3, 3);
  REQUIRE(evaluate(A + A) == device->backend().tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE(evaluate(A + B) == device->backend().tensor<int>({{2,3,4},{2,3,4},{2,3,4}}));

  auto C = device->tensor<int>({1});
  REQUIRE(evaluate(A + C) == device->backend().tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto row = device->tensor<int>({1,2,3});
  auto col = device->tensor<int>({{10},{20},{30}});
  REQUIRE(evaluate(row + col) == device->backend().tensor<int>({{11,12,13},{21,22,23},{31,32,33}}));
}

TEST_CASE( "Softmax is computed", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-7);
  auto A = device->tensor<float>({1,2,3});
  REQUIRE(evaluate(A.softmax()).near_equals(device->backend().tensor<float>({0.09003057, 0.24472847, 0.66524096})));

  auto B =  device->tensor<float>({{1, 0.5, 0.2, 3},{1,  -1,   7, 3},{2,  12,  13, 3}});
  REQUIRE(evaluate(B.softmax()).near_equals(device->backend().tensor<float>({{1.05877070e-01,6.42176889e-02,4.75736340e-02,7.82331607e-01},{2.42746030e-03,3.28521027e-04,9.79307378e-01,1.79366403e-02},{1.22093673e-05,2.68929212e-01,7.31025390e-01,3.31885014e-05}})));
}

TEST_CASE( "Log Softmax backwards", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-5);
  // MAX_TENSOR_LOG_LIMIT = 10000;
  auto W = device->var(device->backend().tensor<float>({1,2,3}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE(program->forward().near_equals(device->backend().tensor<float>({-2.40760596,-1.40760596,-0.40760596})));
  program->backward();
  REQUIRE(W.get_grad().near_equals(device->backend().tensor<float>({-7.15484549,-21.1671683,-59.25661077})));
}

TEST_CASE( "Log Softmax backwards 2", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto W = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6}}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE(program->forward().near_equals(device->backend().tensor<float>({{-2.40760596,-1.40760596,-0.40760596},{-2.40760596,-1.40760596,-0.40760596}})));
  program->backward();
  REQUIRE(W.get_grad().near_equals(device->backend().tensor<float>({{-7.15484549,-21.1671683,-59.25661077},{-162.7944501,-444.23947731,-1209.28638048}})));
}

TEST_CASE( "Mat mul backwards", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto B = device->var(device->backend().tensor<float>({{3,2,1},{6,5,4},{7,8,9}}));
  auto matmul = device->matmul(A, B);
  auto program = device->compile(matmul);
  REQUIRE(program->forward().near_equals(device->backend().tensor<float>({{36,36 ,36},{84,81,78},{132,126,120}})));
  program->backward();
  REQUIRE(A.get_grad() == device->backend().tensor<float>({{6,15,24},{6,15,24},{6,15,24}}));
  REQUIRE(B.get_grad() == device->backend().tensor<float>({{12,12,12},{15,15,15},{18,18,18}}));
}

TEST_CASE( "Opertaions with constant is computed", "[cpu]" ) {
  auto A = device->tensor<int>({1,2,3});
  REQUIRE(evaluate(10 * A) == device->backend().tensor<int>({10,20,30}));
  REQUIRE(evaluate(A * 10) == device->backend().tensor<int>({10,20,30}));
  REQUIRE(evaluate(A + 10) == device->backend().tensor<int>({11,12,13}));
  REQUIRE(evaluate(A.max(2)) == device->backend().tensor<int>({2,2,3}));
  REQUIRE(evaluate(device->max(A, 2)) == device->backend().tensor<int>({2,2,3}));
  REQUIRE(evaluate(device->max(2, A)) == device->backend().tensor<int>({2,2,3}));
  REQUIRE(evaluate(device->max(A, A)) == device->backend().tensor<int>({1,2,3}));
}

TEST_CASE( "Log softmax sum backwards", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto sum1 = device->sum(A, {0});
  auto sf1 = device->log_softmax(sum1);
  auto sum2 = device->sum(A, {1});
  auto sf2 = device->log_softmax(sum2);
  auto program = device->compile(sf1);
  program->forward(), program->backward();
  REQUIRE(A.get_grad().near_equals(device->backend().tensor<float>({{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08},{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08},{-4.88263374e+05,-9.80705112e+06,-1.96979906e+08}})));
  program = device->compile(sf2);
  program->forward(), program->backward();
  REQUIRE(A.get_grad().near_equals(device->backend().tensor<float>({{-1.20928638e+03,-1.20928638e+03,-1.20928638e+03},{-9.80705112e+06,-9.80705112e+06,-9.80705112e+06},{-7.94673664e+10,-7.94673664e+10,-7.94673664e+10}})));
}

TEST_CASE( "Log softmax mean backwards", "[cpu]" ) {
  NEAR_EQUAL_EPS = Constant(1e-4);
  auto A = device->var(device->backend().tensor<float>({{1,2,3},{4,5,6},{7,8,9}}));
  auto mean1 = device->mean(A, {0});
  auto sf1 = device->log_softmax(mean1);
  auto mean2 = device->mean(A, {1});
  auto sf2 = device->log_softmax(mean2);
  auto program = device->compile(sf1);
  REQUIRE(program->forward().near_equals(device->backend().tensor<float>({-2.40760596, -1.40760596, -0.40760596})));
  program->backward();
  REQUIRE(A.get_grad().near_equals(device->backend().tensor<float>({{-54.2648167,-148.07982577,-403.09546016},{-54.2648167,-148.07982577,-403.09546016},{-54.2648167,-148.07982577,-403.09546016}})));
  program = device->compile(sf2);
  REQUIRE(program->forward().near_equals(device->backend().tensor<float>({-6.05094576,-3.05094576,-0.0509457})));
  program->backward();
  REQUIRE(A.get_grad().near_equals(device->backend().tensor<float>({{-7.05572277,-7.05572277,-7.05572277},{-148.07982577,-148.07982577,-148.07982577},{-2980.62465371,-2980.62465371,-2980.62465371}})));
}
