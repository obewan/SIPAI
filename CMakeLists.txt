cmake_minimum_required(VERSION 3.5.0)


option(ENABLE_COVERAGE "Enable coverage reporting for gcc/clang" TRUE)

set(PROJECT_NAME sipai)
set(SOURCE_DIR src)
file(GLOB SOURCE "${SOURCE_DIR}/*.cpp")
set(LIBRARIES_DIR libs)
set(LIBRARIES libsipai)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP /EHsc")
else() 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -pthread -Wall")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug AND NOT WIN32)
    # Add gprof 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")    

    # Add gcov and lcov
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")    
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 --coverage")    
    set(CMAKE_EXE_LINKER_FLAGS "-lgcov")
    add_custom_target(lcov
        COMMAND lcov --capture --directory . --output-file ${CMAKE_CURRENT_SOURCE_DIR}/coverage/lcov.info
        COMMAND lcov --remove ${CMAKE_CURRENT_SOURCE_DIR}/coverage/lcov.info --output-file ${CMAKE_CURRENT_SOURCE_DIR}/coverage/lcov.info 
        '/usr/include/*' '/usr/lib/*' 
        '${CMAKE_CURRENT_SOURCE_DIR}/test/*'
        '${CMAKE_CURRENT_SOURCE_DIR}/libs/libsipai/**Vulkan*' 
        '${CMAKE_CURRENT_SOURCE_DIR}/src/*'
        '${CMAKE_CURRENT_SOURCE_DIR}/libs/libcsvparser/*'
        '${CMAKE_CURRENT_SOURCE_DIR}/libs/libjson/*'
        COMMENT "Generating lcov.info"
    )
    add_custom_target(lcov_summary
        COMMAND lcov --summary ${CMAKE_CURRENT_SOURCE_DIR}/coverage/lcov.info
        COMMENT "Generating lcov summary"
    )
endif()

project(${PROJECT_NAME})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

add_executable(${PROJECT_NAME} ${SOURCE})

foreach(LIBRARY ${LIBRARIES})
    add_subdirectory("${LIBRARIES_DIR}/${LIBRARY}")
endforeach(LIBRARY)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})

if(WIN32)
    target_sources(${PROJECT_NAME} PRIVATE version.rc)
endif()

enable_testing()
add_subdirectory(test)

install(TARGETS ${PROJECT_NAME} 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
