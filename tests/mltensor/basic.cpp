#include "../common.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mltensor/mltensor.hpp>

using namespace katoml;
using namespace katoml::tensor;

auto backend = construct_cpu_backend();

TEST_CASE("[mltensor] Add is computed" ) {
  auto A = backend->ones_i32(3, 3);
  REQUIRE_THAT(A + A, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE_THAT(A + B, EqualsTensor(backend->tensor<int>({{2,3,4},{2,3,4},{2,3,4}})));

  auto C = backend->tensor<int>({1});
  REQUIRE_THAT(A + C, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE_THAT(row + col, EqualsTensor(backend->tensor<int>({{11,12,13},{21,22,23},{31,32,33}})));
}

TEST_CASE("[mltensor] Add assign is computed") {
  auto A = backend->ones_i32(3, 3);
  A += A;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  A = backend->ones_i32(3, 3);
  auto B = backend->tensor<int>({1,2,3});
  A += B;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{2,3,4},{2,3,4},{2,3,4}})));

  A = backend->ones_i32(3, 3);
  auto C = backend->tensor<int>({1});
  A += C;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  row += col;
  REQUIRE_THAT(row, EqualsTensor(backend->tensor<int>({{11,12,13},{21,22,23},{31,32,33}})));
}

TEST_CASE("[mltensor] Sub is computed") {
  auto A = backend->ones_i32(3, 3);
  REQUIRE_THAT(A - A, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE_THAT(A - B, EqualsTensor(backend->tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}})));

  auto C = backend->tensor<int>({1});
  REQUIRE_THAT(A - C, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE_THAT(row - col, EqualsTensor(backend->tensor<int>({{-9,-8,-7},{-19,-18,-17},{-29,-28,-27}})));
}

TEST_CASE("[mltensor] Sub assign is computed") {
  auto A = backend->ones_i32(3, 3);
  A -= A;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));

  A = backend->ones_i32(3, 3);
  auto B = backend->tensor<int>({1,2,3});
  A -= B;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}})));

  A = backend->ones_i32(3, 3);
  auto C = backend->tensor<int>({1});
  A -= C;
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  row -= col;
  REQUIRE_THAT(row, EqualsTensor(backend->tensor<int>({{-9,-8,-7},{-19,-18,-17},{-29,-28,-27}})));
}

TEST_CASE("[mltensor] Mul is computed") {
  auto A = backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}});
  REQUIRE_THAT(A * A, EqualsTensor(backend->tensor<int>({{4,4,4},{4,4,4},{4,4,4}})));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE_THAT(A * B, EqualsTensor(backend->tensor<int>({{2,4,6},{2,4,6},{2,4,6}})));

  auto C = backend->tensor<int>({3});
  REQUIRE_THAT(A * C, EqualsTensor(backend->tensor<int>({{6,6,6},{6,6,6},{6,6,6}})));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE_THAT(row * col, EqualsTensor(backend->tensor<int>({{10,20,30},{20,40,60},{30,60,90}})));
}

TEST_CASE("[mltensor] Less is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE_THAT((A < B), EqualsTensor(backend->tensor<int>({{0,1},{1,0}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE_THAT((A < B), EqualsTensor(backend->tensor<int>({{0,1},{0,0}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{2},{3}});
  REQUIRE_THAT((A < B), EqualsTensor(backend->tensor<int>({{0,0},{1,0}})));
}

TEST_CASE("[mltensor] Less or eq is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,2},{3,2}});
  REQUIRE_THAT((A <= B), EqualsTensor(backend->tensor<int>({{0,1},{1,0}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE_THAT((A <= B), EqualsTensor(backend->tensor<int>({{0,1},{1,0}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{2},{3}});
  REQUIRE_THAT((A <= B), EqualsTensor(backend->tensor<int>({{0,1},{1,0}})));
}

TEST_CASE("[mltensor] More is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE_THAT((A > B), EqualsTensor(backend->tensor<int>({{1,0},{0,1}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE_THAT((A > B), EqualsTensor(backend->tensor<int>({{1,0},{0,1}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{1},{2}});
  REQUIRE_THAT((A > B), EqualsTensor(backend->tensor<int>({{1,1},{0,1}})));
}

TEST_CASE("[mltensor] More or eq is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE_THAT((A >= B), EqualsTensor(backend->tensor<int>({{1,0},{0,1}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE_THAT((A >= B), EqualsTensor(backend->tensor<int>({{1,0},{1,1}})));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{1},{2}});
  REQUIRE_THAT((A >= B), EqualsTensor(backend->tensor<int>({{1,1},{1,1}})));
}

TEST_CASE("[mltensor] Bin ops with constant is computed") {
  auto A = backend->ones_i32(3, 3);
  REQUIRE_THAT(A + 1, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));
  REQUIRE_THAT(1 + A, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));
  REQUIRE_THAT(A - 1, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));
  REQUIRE_THAT(2 - A, EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
  REQUIRE_THAT(A * 32, EqualsTensor(backend->tensor<int>({{32,32,32},{32,32,32},{32,32,32}})));
  REQUIRE_THAT(32 * A, EqualsTensor(backend->tensor<int>({{32,32,32},{32,32,32},{32,32,32}})));
  REQUIRE_THAT(A / 1, EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
  REQUIRE_THAT(1 / A, EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
  REQUIRE_THAT(A / 2, EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));
  REQUIRE_THAT(2 / A, EqualsTensor(backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}})));
  REQUIRE_THAT((A < 1), EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));
  REQUIRE_THAT((1 < A), EqualsTensor(backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}})));
  REQUIRE_THAT((A <= 1), EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
  REQUIRE_THAT((1 <= A), EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
}

TEST_CASE("[mltensor] Matmul is computed") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto B = backend->tensor<int>({{4,5,6},{6,5,4},{4,6,5}});
  std::vector<std::vector<int>> ans = {{28,33,29},{28,31,31},{28,33,29}};
  REQUIRE_THAT(A.matmul(B), EqualsTensor(backend->from_vector<int>(ans)));

  std::vector<std::vector<std::vector<int>>> ans_mul;
  for (int i=0;i<3;i++) ans_mul.push_back(ans);
  auto A_mul = A * backend->ones_i32(3,3,3);
  REQUIRE_THAT(A_mul.matmul(B), EqualsTensor(backend->from_vector<int>(ans_mul)));
}

TEST_CASE("[mltensor] Reshape is working") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,9}));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));

  A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,Shape::Any}));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));

  auto reshaped = A.reshaped(Shape({3,3}));
  REQUIRE_THAT(reshaped, EqualsTensor(backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}})));
}

TEST_CASE("[mltensor] Add assign to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));
  A_view += backend->tensor<int>({1});
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{2,3,4,4,3,2,2,3,4}}})));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{2,3,4},{4,3,2},{2,3,4}})));
  
  auto B = backend->tensor<int>({1});
  auto B_view = B.reshaped(Shape({1,1}));
  REQUIRE_THROWS_AS([&](){
    B_view += A;
  }(), ViewAssignAllocationError);
}

TEST_CASE("[mltensor] Assign to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));
  A_view.assign(backend->tensor<int>({1}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,1,1,1,1,1,1,1,1}}})));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}})));
  
  auto B = backend->tensor<int>({1});
  auto B_view = B.reshaped(Shape({1,1}));
  REQUIRE_THROWS_AS([&](){
    B_view.assign(A);
  }(), ViewAssignAllocationError);
}

TEST_CASE("[mltensor] Div assign to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));
  A_view /= backend->tensor<int>({2});
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{0,1,1,1,1,0,0,1,1}}})));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{0,1,1},{1,1,0},{0,1,1}})));
}

TEST_CASE("[mltensor] Mul assign to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));
  A_view *= backend->tensor<int>({2});
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{2,4,6,6,4,2,2,4,6}}})));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{2,4,6},{6,4,2},{2,4,6}})));
  
  auto B = backend->tensor<int>({1});
  auto B_view = B.reshaped(Shape({1,1}));
  REQUIRE_THROWS_AS([&](){
    B_view *= A;
  }(), ViewAssignAllocationError);
}

TEST_CASE("[mltensor] Fill to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}})));
  A_view.assign(20);
  REQUIRE_THAT(A_view, EqualsTensor(backend->tensor<int>({{{20,20,20,20,20,20,20,20,20}}})));
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{20,20,20},{20,20,20},{20,20,20}})));
}

TEST_CASE("[mltensor] Use freed tensor") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  A = backend->tensor<int>({1});
  
  REQUIRE_THROWS_AS([&](){
    A_view.assign(1);
  }(), UseAfterFreeError);
    
  REQUIRE_THROWS_AS([&](){
    auto C = A_view + 1;
    std::cout << C;
  }(), UseAfterFreeError);

  REQUIRE_THROWS_AS([&](){
    auto C = A_view.copy();
    std::cout << C;
  }(), UseAfterFreeError);

  REQUIRE_THROWS_AS([&](){
    auto C = A_view.reshaped(Shape({3,3}));
    std::cout << C;
  }(), UseAfterFreeError);
}

TEST_CASE("[mltensor] Reduce sum is working") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  REQUIRE(A.sum().at(0).cast<int>() == 18);
  REQUIRE_THAT(A.sum({0}), EqualsTensor(backend->tensor<int>({5,6,7})));
  REQUIRE_THAT(A.sum({1}), EqualsTensor(backend->tensor<int>({6,6,6})));
  REQUIRE_THAT(A.sum({-2}), EqualsTensor(backend->tensor<int>({5,6,7})));
  REQUIRE_THAT(A.sum({-1}), EqualsTensor(backend->tensor<int>({6,6,6})));
}

TEST_CASE("[mltensor] Transpose is working") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{2,1,3}});
  A.transpose();
  REQUIRE_THAT(A, EqualsTensor(backend->tensor<int>({{1, 3, 2}, {2, 2, 1}, {3, 1, 3}})));
  REQUIRE_THAT(A.transposed(), EqualsTensor(backend->tensor<int>({{1,2,3},{3,2,1},{2,1,3}})));
}

TEST_CASE("[mltensor] Reduce mean is working") {
  auto A = backend->tensor<int>({{3,3,3}});
  
  REQUIRE_THAT(A.mean(), EqualsTensor(backend->tensor<int>({3})));

  auto B = backend->tensor<float>({{1,2,3},{4,5,6},{7,8,9}});
  
  REQUIRE_THAT(B.mean({0}), EqualsTensor(backend->tensor<float>({4,5,6})));
  REQUIRE_THAT(B.mean({1}), EqualsTensor(backend->tensor<float>({2,5,8})));
  REQUIRE_THAT(B.mean({-2}), EqualsTensor(backend->tensor<float>({4,5,6})));
  REQUIRE_THAT(B.mean({-1}), EqualsTensor(backend->tensor<float>({2,5,8})));
}

TEST_CASE("[mltensor] at is working") {
  auto A = backend->tensor<int>({{3,3,3}});
  REQUIRE(A.at(0,0).cast<int>() == 3);
  REQUIRE(A.at(0,1).cast<int>() == 3);
  REQUIRE(A.at(0,2).cast<int>() == 3);
  for (int i=0;i<1000;i++){
    REQUIRE(A.at(0,2).cast<int>() == 3+i);
    A.at(0,2).raw<int>() += 1;
  }
}

TEST_CASE("[mltensor] index is working") {
  auto A = backend->tensor<int>({{3,3,3}});
  REQUIRE(A(0,0).cast<int>() == 3);
  REQUIRE(A(0,1).cast<int>() == 3);
  REQUIRE(A(0,2).cast<int>() == 3);
  for (int i=0;i<1000;i++){
    REQUIRE(A(0,2).cast<int>() == 3+i);
    A(0,2).raw<int>() += 1;
  }
}

TEST_CASE("[mltensor] at_typed is working") {
  auto A = backend->tensor<int>({{1,2,(int)1e9}});

  REQUIRE(A.at_typed<int>(0,0) == 1);
  REQUIRE(A.at_typed<int>(0,1) == 2);
  REQUIRE(A.at_typed<int>(0,2) == (int)1e9);
  for (int i=0;i<1000;i++){
    REQUIRE(A.at_typed<int>(0,2) == (int)1e9+i);
    A.at_typed<int>(0,2) += 1;
  }
}

TEST_CASE("[mltensor] index arbitrary constant is working") {
  auto A = backend->tensor<int>({{1,2,(int)1e9}});

  A(0,0) = 0;

  REQUIRE(A(0,0).cast<int>() == 0);
  REQUIRE(A(0,1).cast<int>() == 2);
  REQUIRE(A(0,2).cast<int>() == (int)1e9);

  auto B = backend->tensor<float>({0.0f});
  B(0) = 1.5f;
  REQUIRE(B(0).cast<float>() == 1.5f);
}

TEST_CASE("[mltensor] save and load is working") {
  auto A = backend->tensor<int>({{1,2,(int)1e9}});
  auto buffer = A.save();
  auto B = backend->load(buffer);
  REQUIRE_THAT(A, EqualsTensor(B));
  auto A_transposed = A.transposed();
  auto C = backend->load(A_transposed.save());
  REQUIRE_THAT(A_transposed, EqualsTensor(C));
}
