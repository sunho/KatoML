#include "katoml/mltensor/core.hpp"
#include "katoml/mltensor/errors.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mltensor/mltensor.hpp>

using namespace katoml::tensor;

auto backend = construct_cpu_backend();

TEST_CASE("[mltensor] Add is computed" ) {
  auto A = backend->ones_i32(3, 3);
  REQUIRE(A + A == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE(A + B == backend->tensor<int>({{2,3,4},{2,3,4},{2,3,4}}));

  auto C = backend->tensor<int>({1});
  REQUIRE(A + C == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE(row + col == backend->tensor<int>({{11,12,13},{21,22,23},{31,32,33}}));
}

TEST_CASE("[mltensor] Add assign is computed") {
  auto A = backend->ones_i32(3, 3);
  A += A;
  REQUIRE(A == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  A = backend->ones_i32(3, 3);
  auto B = backend->tensor<int>({1,2,3});
  A += B;
  REQUIRE(A == backend->tensor<int>({{2,3,4},{2,3,4},{2,3,4}}));

  A = backend->ones_i32(3, 3);
  auto C = backend->tensor<int>({1});
  A += C;
  REQUIRE(A == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  row += col;
  REQUIRE(row == backend->tensor<int>({{11,12,13},{21,22,23},{31,32,33}}));
}

TEST_CASE("[mltensor] Sub is computed") {
  auto A = backend->ones_i32(3, 3);
  REQUIRE(A - A == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE(A - B == backend->tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}}));

  auto C = backend->tensor<int>({1});
  REQUIRE(A - C == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE(row - col == backend->tensor<int>({{-9,-8,-7},{-19,-18,-17},{-29,-28,-27}}));
}

TEST_CASE("[mltensor] Sub assign is computed") {
  auto A = backend->ones_i32(3, 3);
  A -= A;
  REQUIRE(A == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  A = backend->ones_i32(3, 3);
  auto B = backend->tensor<int>({1,2,3});
  A -= B;
  REQUIRE(A == backend->tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}}));

  A = backend->ones_i32(3, 3);
  auto C = backend->tensor<int>({1});
  A -= C;
  REQUIRE(A == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  row -= col;
  REQUIRE(row == backend->tensor<int>({{-9,-8,-7},{-19,-18,-17},{-29,-28,-27}}));
}

TEST_CASE("[mltensor] Mul is computed") {
  auto A = backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}});
  REQUIRE(A * A == backend->tensor<int>({{4,4,4},{4,4,4},{4,4,4}}));

  auto B = backend->tensor<int>({1,2,3});
  REQUIRE(A * B == backend->tensor<int>({{2,4,6},{2,4,6},{2,4,6}}));

  auto C = backend->tensor<int>({3});
  REQUIRE(A * C == backend->tensor<int>({{6,6,6},{6,6,6},{6,6,6}}));

  auto row = backend->tensor<int>({1,2,3});
  auto col = backend->tensor<int>({{10},{20},{30}});
  REQUIRE(row * col == backend->tensor<int>({{10,20,30},{20,40,60},{30,60,90}}));
}

TEST_CASE("[mltensor] Less is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE((A < B) == backend->tensor<int>({{0,1},{1,0}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE((A < B) == backend->tensor<int>({{0,1},{0,0}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{2},{3}});
  REQUIRE((A < B) == backend->tensor<int>({{0,0},{1,0}}));
}

TEST_CASE("[mltensor] Less or eq is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,2},{3,2}});
  REQUIRE((A <= B) == backend->tensor<int>({{0,1},{1,0}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE((A <= B) == backend->tensor<int>({{0,1},{1,0}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{2},{3}});
  REQUIRE((A <= B) == backend->tensor<int>({{0,1},{1,0}}));
}

TEST_CASE("[mltensor] More is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE((A > B) == backend->tensor<int>({{1,0},{0,1}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE((A > B) == backend->tensor<int>({{1,0},{0,1}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{1},{2}});
  REQUIRE((A > B) == backend->tensor<int>({{1,1},{0,1}}));
}

TEST_CASE("[mltensor] More or eq is computed") {
  auto A = backend->tensor<int>({{4,2},{2,4}});
  auto B = backend->tensor<int>({{2,3},{3,2}});
  REQUIRE((A >= B) == backend->tensor<int>({{1,0},{0,1}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({2,3});
  REQUIRE((A >= B) == backend->tensor<int>({{1,0},{1,1}}));

  A = backend->tensor<int>({{4,2},{2,4}});
  B = backend->tensor<int>({{1},{2}});
  REQUIRE((A >= B) == backend->tensor<int>({{1,1},{1,1}}));
}

TEST_CASE("[mltensor] Bin ops with constant is computed") {
  auto A = backend->ones_i32(3, 3);
  REQUIRE(A + 1 == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));
  REQUIRE(1 + A == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));
  REQUIRE(A - 1 == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));
  REQUIRE(2 - A == backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}}));
  REQUIRE(A * 32 == backend->tensor<int>({{32,32,32},{32,32,32},{32,32,32}}));
  REQUIRE(32 * A == backend->tensor<int>({{32,32,32},{32,32,32},{32,32,32}}));
  REQUIRE(A / 1 == backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}}));
  REQUIRE(1 / A == backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}}));
  REQUIRE(A / 2 == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));
  REQUIRE(2 / A == backend->tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));
  REQUIRE((A < 1) == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));
  REQUIRE((1 < A) == backend->tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));
  REQUIRE((A <= 1) == backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}}));
  REQUIRE((1 <= A) == backend->tensor<int>({{1,1,1},{1,1,1},{1,1,1}}));
}

TEST_CASE("[mltensor] Matmul is computed") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto B = backend->tensor<int>({{4,5,6},{6,5,4},{4,6,5}});
  std::vector<std::vector<int>> ans = {{28,33,29},{28,31,31},{28,33,29}};
  REQUIRE(A.matmul(B) == backend->from_vector<int>(ans));

  std::vector<std::vector<std::vector<int>>> ans_mul;
  for (int i=0;i<3;i++) ans_mul.push_back(ans);
  auto A_mul = A * backend->ones_i32(3,3,3);
  REQUIRE(A_mul.matmul(B) == backend->from_vector<int>(ans_mul));
}

TEST_CASE("[mltensor] Reshape is working") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,9}));
  REQUIRE(A == backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));

  A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,Shape::Any}));
  REQUIRE(A == backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));

  auto reshaped = A.reshaped(Shape({3,3}));
  REQUIRE(reshaped == backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}}));
}

TEST_CASE("[mltensor] Add assign to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE(A_view == backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));
  A_view += backend->tensor<int>({1});
  REQUIRE(A_view == backend->tensor<int>({{{2,3,4,4,3,2,2,3,4}}}));
  REQUIRE(A == backend->tensor<int>({{2,3,4},{4,3,2},{2,3,4}}));
  
  auto B = backend->tensor<int>({1});
  auto B_view = B.reshaped(Shape({1,1}));
  REQUIRE_THROWS_AS([&](){
    B_view += A;
  }(), ViewAssignAllocationError);
}

TEST_CASE("[mltensor] Fill to view") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  REQUIRE(A_view == backend->tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));
  A_view.fill(20);
  REQUIRE(A_view == backend->tensor<int>({{{20,20,20,20,20,20,20,20,20}}}));
  REQUIRE(A == backend->tensor<int>({{20,20,20},{20,20,20},{20,20,20}}));
}

TEST_CASE("[mltensor] Use freed tensor") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto A_view = A.reshaped(Shape({1,1,9}));
  A = backend->tensor<int>({1});
  
  REQUIRE_THROWS_AS([&](){
    A_view.fill(1);
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
  REQUIRE(A.sum().at<int>(0) == 18);
  REQUIRE(A.sum({0}) == backend->tensor<int>({5,6,7}));
  REQUIRE(A.sum({1}) == backend->tensor<int>({6,6,6}));
  REQUIRE(A.sum({-2}) == backend->tensor<int>({5,6,7}));
  REQUIRE(A.sum({-1}) == backend->tensor<int>({6,6,6}));
}

TEST_CASE("[mltensor] Transpose is working") {
  auto A = backend->tensor<int>({{1,2,3},{3,2,1},{2,1,3}});
  A.transpose();
  REQUIRE(A == backend->tensor<int>({{1, 3, 2}, {2, 2, 1}, {3, 1, 3}}));
  REQUIRE(A.transposed() ==  backend->tensor<int>({{1,2,3},{3,2,1},{2,1,3}}));
}

TEST_CASE("[mltensor] Reduce mean is working") {
  auto A = backend->tensor<int>({{3,3,3}});
  
  REQUIRE(A.mean() == backend->tensor<int>({3}));

  auto B = backend->tensor<float>({{1,2,3},{4,5,6},{7,8,9}});
  
  REQUIRE(B.mean({0}) == backend->tensor<float>({4,5,6}));
  REQUIRE(B.mean({1}) == backend->tensor<float>({2,5,8}));
  REQUIRE(B.mean({-2}) == backend->tensor<float>({4,5,6}));
  REQUIRE(B.mean({-1}) == backend->tensor<float>({2,5,8}));
}
