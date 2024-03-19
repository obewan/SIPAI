cmake_minimum_required(VERSION 3.5.0)

file(GLOB TEST_DATA_FILES "${CMAKE_CURRENT_SOURCE_DIR}/data/*")
foreach(DATA_FILE ${TEST_DATA_FILES})
    get_filename_component(FILENAME ${DATA_FILE} NAME)
    configure_file(${DATA_FILE}
                   ${CMAKE_CURRENT_BINARY_DIR}/data/${FILENAME}
                   COPYONLY)
endforeach()

add_subdirectory(ctests)
add_subdirectory(doctests)