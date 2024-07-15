cmake_minimum_required(VERSION 3.5.0)

#### CTEST - INTEGRATION TESTS
include(CTest) 
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

add_test(NAME test_version COMMAND sipai-cmd -v WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
set_tests_properties(test_version PROPERTIES
  PASS_REGULAR_EXPRESSION "SIPAI - Simple Image Processing Artificial Intelligence"
)