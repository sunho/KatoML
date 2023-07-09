#include "../common.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mlcompiler/mlcompiler.hpp>

using namespace katoml::compiler;
using namespace katoml::tensor;

static auto device = construct_device();

#define evaluate(eval) (device->compile(eval)->forward())

#define TS(typ) device->backend().tensor<typ>

TEST_CASE("[mlcompiler] Add is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A + A), EqualsTensor(TS(int)({{2,2,2},{2,2,2},{2,2,2}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A + B), EqualsTensor(TS(int)({{2,3,4},{2,3,4},{2,3,4}})));

  auto C = device->tensor<int>({1});
  REQUIRE_THAT(evaluate(A + C), EqualsTensor(TS(int)({{2,2,2},{2,2,2},{2,2,2}})));

  auto row = device->tensor<int>({1,2,3});
  auto col = device->tensor<int>({{10},{20},{30}});
  REQUIRE_THAT(evaluate(row + col), EqualsTensor(TS(int)({{11,12,13},{21,22,23},{31,32,33}})));
}

TEST_CASE("[mlcompiler] Sub is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A - A), EqualsTensor(TS(int)({{0,0,0},{0,0,0},{0,0,0}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A - B), EqualsTensor(TS(int)({{0,-1,-2},{0,-1,-2},{0,-1,-2}})));
}

TEST_CASE("[mlcompiler] Mul is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A * A), EqualsTensor(TS(int)({{1,1,1},{1,1,1},{1,1,1}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A * B), EqualsTensor(TS(int)({{1,2,3},{1,2,3},{1,2,3}})));
}

TEST_CASE("[mlcompiler] Div is computed") {
  auto A = device->ones_i32(3, 3);
  REQUIRE_THAT(evaluate(A / A), EqualsTensor(TS(int)({{1,1,1},{1,1,1},{1,1,1}})));

  auto B = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(A / B), EqualsTensor(TS(int)({{1,0,0},{1,0,0},{1,0,0}})));
}

TEST_CASE("[mlcompiler] Softmax is computed 1") {
  auto A = device->tensor<float>({1,2,3});
  REQUIRE_THAT(evaluate(A.softmax()), ApproxTensor(TS(float)({0.09003057, 0.24472847, 0.66524096})));

  auto B =  device->tensor<float>({{1, 0.5, 0.2, 3},{1,  -1,   7, 3},{2,  12,  13, 3}});
  REQUIRE_THAT(evaluate(B.softmax()), ApproxTensor(TS(float)({{1.05877070e-01,6.42176889e-02,4.75736340e-02,7.82331607e-01},{2.42746030e-03,3.28521027e-04,9.79307378e-01,1.79366403e-02},{1.22093673e-05,2.68929212e-01,7.31025390e-01,3.31885014e-05}})));
}

TEST_CASE("[mlcompiler] Exp is computed") {
  auto A = device->var(TS(float)({0.6, 1.0}));
  auto res = device->exp(A);
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({1.8221, 2.7183})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({1.8221, 2.7183})));
}

TEST_CASE("[mlcompiler] Div is computed 1") {
  auto A = device->var(TS(float)({42.0, 1.0}));
  auto B = device->var(TS(float)({2.0, 4.0}));
  auto res = (Node)A / B;
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({21.0000,  0.2500})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({0.5000, 0.2500})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({-10.5000, -0.0625})));
}

TEST_CASE("[mlcompiler] Div is computed 2") {
  auto A = device->var(TS(float)({1.0}));
  auto B = device->var(TS(float)({2.0, 4.0}));
  auto res = (Node)A / B;
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({0.5000, 0.2500})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({0.7500})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({-0.2500, -0.0625})));
}

TEST_CASE("[mlcompiler] Div is computed 3") {
  auto A = device->var(TS(float)({2., 4.}));
  auto B = device->var(TS(float)({2.}));
  auto res = (Node)A / B;
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({1., 2.})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({0.5000, 0.5000})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({-1.5000})));
}

TEST_CASE("[mlcompiler] Div is computed 4") {
  auto A = device->var(TS(float)({{42.}, {1.}}));
  auto B = device->var(TS(float)({2.,5.}));
  auto res = (Node)A / B;
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({{21., 8.4},{0.5000, 0.2000}})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{0.7000},{0.7000}})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({-10.7500, -1.7200})));
}

TEST_CASE("[mlcompiler] Div is computed 5") {
  auto A = device->var(TS(float)({{42., 10.0},{3.0,  0.5}}));
  auto B = device->var(TS(float)({2.,5.}));
  auto res = (Node)A / B;
  REQUIRE_THAT(evaluate(res),  ApproxTensor(TS(float)({{21.0,2.0},{1.5,0.1}})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{0.5,0.2},{0.5,0.2}})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({-11.2500, -0.4200})));
}

TEST_CASE("[mlcompiler] Log softmax is computed fuzzed") {
  auto A = device->var(TS(float)({{0.6000000, 1.0000000},{0.1000000, 0.4000000}}));
  auto B = device->var(TS(float)({{-0.3007284, -0.1314884},{1.6309785,  0.0064734}}));
  auto res = B*device->log_softmax(A);
  REQUIRE_THAT(evaluate(res), ApproxTensor(TS(float)({{0.2745696, 0.0674555},{-1.3934350, -0.0035886}})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{-0.1272745, 0.1272745},{0.9341486, -0.9341485}})));
}

TEST_CASE("[mlcompiler] Softmax is computed fuzzed") {
  auto A = device->var(TS(float)({{0.6000000, 1.0000000},{0.1000000, 0.4000000}}));
  auto B = device->var(TS(float)({{-0.4280916, 0.0129038},{0.8369727, 0.0764691}}));
  auto res = B*device->softmax(A);
  REQUIRE_THAT(evaluate(res), ApproxTensor(TS(float)({{-0.1717984, 0.0077253},{0.3561800, 0.0439271}})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{-0.1059539, 0.1059539},{0.1859114, -0.1859114}})));
}

TEST_CASE("[mlcompiler] Sum is computed fuzzed") {
  auto A = device->var(TS(float)({42.,  1., 53., 20., 10.}));
  auto B = device->var(TS(float)({1.5409961, -0.2934289, -2.1787894,  0.5684313, -1.0845224}));
  auto res = B*device->sum(A);
  REQUIRE_THAT(evaluate(res), ApproxTensor(TS(float)({194.1655121,  -36.9720421, -274.5274658, 71.6223373, -136.6498108})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({-1.4473134, -1.4473134, -1.4473134, -1.4473134, -1.4473134})));
}

TEST_CASE("[mlcompiler] Mean is computed fuzzed") {
  auto A = device->var(TS(float)({42.,  1., 53., 20., 10.}));
  auto B = device->var(TS(float)({1.5409961, -0.2934289, -2.1787894,  0.5684313, -1.0845224}));
  auto res = B*device->mean(A);
  REQUIRE_THAT(evaluate(res), ApproxTensor(TS(float)({38.8331032,  -7.3944082, -54.9054947,  14.3244686, -27.3299637})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({-0.2894627, -0.2894627, -0.2894627, -0.2894627, -0.2894627})));
}

TEST_CASE("[mlcompiler] Matmul is computed fuzzed") {
  auto A = device->var(TS(float)({{1.5409961, -0.2934289, -2.1787894},{0.5684313, -1.0845224, -1.3985955}}));
  auto B = device->var(TS(float)({{0.4033468,  0.8380263},{-0.7192576, -0.4033435},{-0.5966353,  0.1820365}}));
  auto res = -0.8566746*device->matmul(A,B);
  REQUIRE_THAT(evaluate(res), ApproxTensor(TS(float)({{-1.8269011, -0.8679217},{-1.5795172, -0.5647200}})));
  auto program = device->compile(res);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{-1.0634530, 0.9617038, 0.3551763},{-1.0634530,  0.9617038, 0.3551763}})));
  REQUIRE_THAT(B.get_grad(), ApproxTensor(TS(float)({{-1.8070929, -1.8070929},{1.1804558, 1.1804558},{3.0646548, 3.0646548}})));
}

TEST_CASE("[mlcompiler] Log Softmax backwards") {
  auto W = device->var(TS(float)({1,2,3}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE_THAT(program->forward(), ApproxTensor(TS(float)({-2.40760596,-1.40760596,-0.40760596})));
  program->backward();
  REQUIRE_THAT(W.get_grad(), ApproxTensor(TS(float)({0.729908, 0.265815, -0.995723})));
}

TEST_CASE("[mlcompiler] Log Softmax backwards 2") {
  auto W = device->var(TS(float)({{1,2,3},{4,5,6}}));
  auto sf = device->log_softmax(W);
  auto program = device->compile(sf);
  REQUIRE_THAT(program->forward(), ApproxTensor(TS(float)({{-2.40760596,-1.40760596,-0.40760596},{-2.40760596,-1.40760596,-0.40760596}})));
  program->backward();
  REQUIRE_THAT(W.get_grad(), ApproxTensor(TS(float)({{0.729908, 0.265815, -0.995723},{0.729908, 0.265815, -0.995723}})));
}

TEST_CASE("[mlcompiler] Mat mul backwards") {
  auto A = device->var(TS(float)({{1,2,3},{4,5,6},{7,8,9}}));
  auto B = device->var(TS(float)({{3,2,1},{6,5,4},{7,8,9}}));
  auto matmul = device->matmul(A, B);
  auto program = device->compile(matmul);
  REQUIRE_THAT(program->forward(), ApproxTensor(TS(float)({{36,36 ,36},{84,81,78},{132,126,120}})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), EqualsTensor(TS(float)({{6,15,24},{6,15,24},{6,15,24}})));
  REQUIRE_THAT(B.get_grad(), EqualsTensor(TS(float)({{12,12,12},{15,15,15},{18,18,18}})));
}

TEST_CASE("[mlcompiler] Opertaions with constant is computed") {
  auto A = device->tensor<int>({1,2,3});
  REQUIRE_THAT(evaluate(10 * A), EqualsTensor(TS(int)({10,20,30})));
  REQUIRE_THAT(evaluate(A * 10), EqualsTensor(TS(int)({10,20,30})));
  REQUIRE_THAT(evaluate(A + 10), EqualsTensor(TS(int)({11,12,13})));
  REQUIRE_THAT(evaluate(A.max(2)), EqualsTensor(TS(int)({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(A, 2)), EqualsTensor(TS(int)({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(2, A)), EqualsTensor(TS(int)({2,2,3})));
  REQUIRE_THAT(evaluate(device->max(A, A)), EqualsTensor(TS(int)({1,2,3})));
}

TEST_CASE("[mlcompiler] Log softmax sum backwards") {
  auto A = device->var(TS(float)({{1,2,3},{4,5,6},{7,8,9}}));
  auto sum1 = device->sum(A, {0});
  auto sf1 = device->log_softmax(sum1);
  auto sum2 = device->sum(A, {1});
  auto sf2 = device->log_softmax(sum2);
  auto program = device->compile(sf1);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{0.992933, 0.858057, -1.85099},{0.992933, 0.858057, -1.85099},{0.992933, 0.858057, -1.85099}})));
  program = device->compile(sf2);
  program->forward(), program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{1,1,1},{0.99963, 0.99963, 0.99963},{-1.99963, -1.99963, -1.99963}})));
}

TEST_CASE("[mlcompiler] Log softmax mean backwards") {
  auto A = device->var(TS(float)({{1,2,3},{4,5,6},{7,8,9}}));
  auto mean1 = device->mean(A, {0});
  auto sf1 = device->log_softmax(mean1);
  auto mean2 = device->mean(A, {1});
  auto sf2 = device->log_softmax(mean2);
  auto program = device->compile(sf1);
  REQUIRE_THAT(program->forward(), ApproxTensor(TS(float)({-2.40760596, -1.40760596, -0.40760596})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{0.243303, 0.0886048, -0.331908},{0.243303, 0.0886048, -0.331908},{0.243303, 0.0886048, -0.331908}})));
  program = device->compile(sf2);
  REQUIRE_THAT(program->forward(), ApproxTensor(TS(float)({-6.05094576,-3.05094576,-0.0509457})));
  program->backward();
  REQUIRE_THAT(A.get_grad(), ApproxTensor(TS(float)({{0.330978, 0.330978, 0.330978},{0.286019, 0.286019, 0.286019},{-0.616997,-0.616997, -0.6169977}})));
}
