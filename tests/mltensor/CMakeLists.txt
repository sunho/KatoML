file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/**")
add_executable(mltensor_tests ${testSources})
catch_discover_tests(mltensor_tests)
target_link_libraries(mltensor_tests PRIVATE Catch2::Catch2WithMain mltensor)