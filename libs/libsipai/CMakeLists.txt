cmake_minimum_required(VERSION 3.5.0)

option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" TRUE)

if(ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 --coverage")    
    set(CMAKE_EXE_LINKER_FLAGS "-lgcov") 
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread -Wall")


add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libcsvparser csv-parser)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libjson nlohmann-json)


set(PROJECT_NAME libsipai)
set(LIBRARY_NAME libsipai)
set(LIBRARY_HEADERS_DIR include)
file(GLOB LIBRARY_HEADER_FILES ${LIBRARY_HEADERS_DIR}/**.h)
set(LIBRARY_HEADERS ${LIBRARY_HEADERS_FILES})

set(LIBRARY_SOURCE_DIR src)
file(GLOB LIBRARY_SOURCE_FILES ${LIBRARY_SOURCE_DIR}/**.cpp)
set(LIBRARY_SOURCE ${LIBRARY_SOURCE_FILES})

project(${PROJECT_NAME})

add_library(${LIBRARY_NAME} STATIC
    ${LIBRARY_HEADERS}
    ${LIBRARY_SOURCE}
)

target_link_libraries(libsipai ${OpenCV_LIBS} tbb)


target_include_directories(${LIBRARY_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../libcsvparser/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../libjson/include>
    $<INSTALL_INTERFACE:include>
)
