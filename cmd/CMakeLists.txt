cmake_minimum_required(VERSION 3.10.0)

set(PROJECT_NAME sipai-cmd)

project(${PROJECT_NAME})

file(GLOB SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(LIBS_DIR ../libs)
set(LIBSIPAI ${LIBS_DIR}/libsipai)

include_directories(${LIBSIPAI}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(${PROJECT_NAME} ${SOURCE})

target_link_libraries(${PROJECT_NAME} libsipai)

if(WIN32)
    target_sources(${PROJECT_NAME} PRIVATE ../version.rc)
endif()

#copy to parent dir
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_BINARY_DIR})

install(TARGETS ${PROJECT_NAME} 
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)