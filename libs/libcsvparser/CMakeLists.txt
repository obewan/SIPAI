cmake_minimum_required(VERSION 3.5.0)
project(csv_parser)

set(LIBRARY_HEADERS_DIR include)
file(GLOB LIBRARY_HEADER_FILES ${LIBRARY_HEADERS_DIR}/*.h)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${LIBRARY_HEADERS_DIR})