cmake_minimum_required(VERSION 3.5.0)

include(copy_directory.cmake)
copy_directory("${CMAKE_SOURCE_DIR}/data" "${CMAKE_BINARY_DIR}/data")


add_subdirectory(ctests)
add_subdirectory(doctests)