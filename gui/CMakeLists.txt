cmake_minimum_required(VERSION 3.5.0)

set(PROJECT_NAME sipai-gui)

project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets)

#rq. Need to add the .h as a source because QT moc will create a cpp from the .h (Q_OBJECT), 
# or else put the .h in same directory than the .cpp to avoid link issues
file(GLOB_RECURSE SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

set(LIBS_DIR ../libs)
set(LIBSIPAI ${LIBS_DIR}/libsipai)

include_directories(${LIBSIPAI}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

qt_standard_project_setup()

qt_add_executable(${PROJECT_NAME} MANUAL_FINALIZATION ${SOURCE})

if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE TRUE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MP /EHsc /Zc:__cplusplus")
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SRCDIR="${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt::Gui Qt6::Widgets)

if(WIN32)
    target_sources(${PROJECT_NAME} PRIVATE ../version.rc)
endif()

qt_finalize_executable(${PROJECT_NAME})

#copy to parent dir
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/icons DESTINATION ${CMAKE_BINARY_DIR}/)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
     COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_BINARY_DIR})

     
install(TARGETS ${PROJECT_NAME} 
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static)

