file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/**")
add_executable(mlcompiler_tests ${testSources})
catch_discover_tests(mlcompiler_tests)
target_link_libraries(mlcompiler_tests PRIVATE Catch2::Catch2WithMain mlcompiler)