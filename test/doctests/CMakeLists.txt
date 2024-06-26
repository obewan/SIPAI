cmake_minimum_required(VERSION 3.5.0)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /EHsc")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    if(NOT WIN32)
        # Add coverage
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")    
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 --coverage")    
        set(CMAKE_EXE_LINKER_FLAGS "-lgcov")
    endif()
        
    # Add libs
    target_sources(nlohmann-json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libjson/include/json.hpp)
    add_library(lib_doctest INTERFACE)
    target_sources(lib_doctest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/doctest.h)
    set(LIBRARY_HEADERS_DIR include)
    target_include_directories(lib_doctest INTERFACE  "${CMAKE_CURRENT_SOURCE_DIR}/${LIBRARY_HEADERS_DIR}")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libsipai/include)
    
    # Create the unit tester
    file(GLOB SOURCES "*.cpp")
    add_executable(tester ${SOURCES}) 
    target_link_libraries(tester PRIVATE lib_doctest nlohmann-json libsipai)    
endif()
