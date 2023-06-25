#include "katoml/mltensor/core.hpp"
#include <catch2/catch_test_macros.hpp>
#include <katoml/mltensor/mltensor.hpp>

using namespace katoml::tensor;

auto backend = construct_cpu_backend();

TEST_CASE( "Add is computed", "[cpu]" ) {
  auto A = backend.ones_i32(3, 3);
  REQUIRE(A + A == backend.tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto B = backend.tensor<int>({1,2,3});
  REQUIRE(A + B == backend.tensor<int>({{2,3,4},{2,3,4},{2,3,4}}));

  auto C = backend.tensor<int>({1});
  REQUIRE(A + C == backend.tensor<int>({{2,2,2},{2,2,2},{2,2,2}}));

  auto row = backend.tensor<int>({1,2,3});
  auto col = backend.tensor<int>({{10},{20},{30}});
  REQUIRE(row + col == backend.tensor<int>({{11,12,13},{21,22,23},{31,32,33}}));
}

TEST_CASE( "Sub is computed", "[cpu]" ) {
  auto A = backend.ones_i32(3, 3);
  REQUIRE(A - A == backend.tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  auto B = backend.tensor<int>({1,2,3});
  REQUIRE(A - B == backend.tensor<int>({{0,-1,-2},{0,-1,-2},{0,-1,-2}}));

  auto C = backend.tensor<int>({1});
  REQUIRE(A - C == backend.tensor<int>({{0,0,0},{0,0,0},{0,0,0}}));

  auto row = backend.tensor<int>({1,2,3});
  auto col = backend.tensor<int>({{10},{20},{30}});
  REQUIRE(row - col == backend.tensor<int>({{-9,-8,-7},{-19,-18,-17},{-29,-28,-27}}));
}

TEST_CASE( "Mul is computed", "[cpu]" ) {
  auto A = backend.tensor<int>({{2,2,2},{2,2,2},{2,2,2}});
  REQUIRE(A * A == backend.tensor<int>({{4,4,4},{4,4,4},{4,4,4}}));

  auto B = backend.tensor<int>({1,2,3});
  REQUIRE(A * B == backend.tensor<int>({{2,4,6},{2,4,6},{2,4,6}}));

  auto C = backend.tensor<int>({3});
  REQUIRE(A * C == backend.tensor<int>({{6,6,6},{6,6,6},{6,6,6}}));

  auto row = backend.tensor<int>({1,2,3});
  auto col = backend.tensor<int>({{10},{20},{30}});
  REQUIRE(row * col == backend.tensor<int>({{10,20,30},{20,40,60},{30,60,90}}));
}

TEST_CASE( "Matmul is computed", "[cpu]" ) {
  auto A = backend.tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  auto B = backend.tensor<int>({{4,5,6},{6,5,4},{4,6,5}});
  std::vector<std::vector<int>> ans = {{28,33,29},{28,31,31},{28,33,29}};
  REQUIRE(A.matmul(B) == backend.tensor<int>(ans));

  std::vector<std::vector<std::vector<int>>> ans_mul;
  for (int i=0;i<3;i++) ans_mul.push_back(ans);
  auto A_mul = A * backend.ones_i32(3,3,3);
  REQUIRE(A_mul.matmul(B) == backend.tensor<int>(ans_mul));
}

TEST_CASE( "Reshape is working", "[cpu]" ) {
  auto A = backend.tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,9}));
  REQUIRE(A == backend.tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));

  A = backend.tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  A.reshape(Shape({1,1,Shape::Any}));
  REQUIRE(A == backend.tensor<int>({{{1,2,3,3,2,1,1,2,3}}}));
}

TEST_CASE( "Reduce sum is working", "[cpu]" ) {
  auto A = backend.tensor<int>({{1,2,3},{3,2,1},{1,2,3}});
  REQUIRE(A.sum().at<int>(0) == 18);
  REQUIRE(A.sum({0}) == backend.tensor<int>({5,6,7}));
  REQUIRE(A.sum({1}) == backend.tensor<int>({6,6,6}));
}

TEST_CASE( "Transpose is working", "[cpu]" ) {
  auto A = backend.tensor<int>({{1,2,3},{3,2,1},{2,1,3}});
  A.transpose();
  REQUIRE(A == backend.tensor<int>({{1, 3, 2}, {2, 2, 1}, {3, 1, 3}}));
}

TEST_CASE( "Mean is working", "[cpu]" ) {
  auto A = backend.tensor<int>({{3,3,3}});
  
  REQUIRE(A.mean() == backend.tensor<int>({3}));

  auto B = backend.tensor<float>({{1,2,3},{4,5,6},{7,8,9}});
  
  REQUIRE(B.mean({0}) == backend.tensor<float>({4,5,6}));
  REQUIRE(B.mean({1}) == backend.tensor<float>({2,5,8}));
}
