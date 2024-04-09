cmake_minimum_required(VERSION 3.5.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    # Add coverage
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 --coverage")    
    set(CMAKE_EXE_LINKER_FLAGS "-lgcov")
    
    # Add libs
    target_sources(nlohmann-json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libjson/include/json.hpp)
    add_library(lib_doctest INTERFACE)
    target_sources(lib_doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/doctest.h)
    set(LIBRARY_HEADERS_DIR include)
    target_include_directories(lib_doctest INTERFACE  "${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_HEADERS_DIR}")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libsipai/include)
    file(GLOB SOURCES "*.cpp")

    # Create the unit tester
    add_executable(tester ${SOURCES}) 
    target_link_libraries(tester PRIVATE ${LIBRARY_NAME} lib_doctest nlohmann-json libsipai)
endif()
