cmake_minimum_required(VERSION 3.10.0)

set(PROJECT_NAME tester)

project(${PROJECT_NAME})

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
    
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libsipai/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../libs/libjson/include)
    
    # Create the unit tester
    file(GLOB SOURCES "*.cpp")
    add_executable(${PROJECT_NAME} ${SOURCES}) 
    target_link_libraries(${PROJECT_NAME} PRIVATE libsipai)    
endif()
