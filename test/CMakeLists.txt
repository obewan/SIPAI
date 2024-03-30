cmake_minimum_required(VERSION 3.5.0)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ctests/images-test1.csv DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/doctests/images-test1.csv DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/doctests/)

add_subdirectory(ctests)
add_subdirectory(doctests)