file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/**")
add_executable(mlapp_tests ${testSources})
catch_discover_tests(mlapp_tests)
target_link_libraries(mlapp_tests PRIVATE Catch2::Catch2WithMain mlapp)