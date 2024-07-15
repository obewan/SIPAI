cmake_minimum_required(VERSION 3.5.0)

# set(PROJECT_NAME sipai-gui)

# find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# project(${PROJECT_NAME})

# file(GLOB SOURCES "*.cpp")

# add_executable(${PROJECT_NAME} ${SOURCE})
# target_link_libraries(${PROJECT_NAME} PRIVATE Qt6::Core Qt6::Widgets)

#copy to parent dir
# add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_BINARY_DIR})