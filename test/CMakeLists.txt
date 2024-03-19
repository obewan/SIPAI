cmake_minimum_required(VERSION 3.5.0)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

add_subdirectory(ctests)
add_subdirectory(doctests)