# Copy the source directory to the destination directory
function(copy_directory source_dir destination_dir)
    # Create the destination directory
    file(MAKE_DIRECTORY "${destination_dir}")

    # Get the list of files and directories in the source directory
    file(GLOB_RECURSE files_list "${source_dir}/*")

    # Copy each file and directory to the destination directory
    foreach(file ${files_list})
        if(IS_DIRECTORY "${file}")
            # Create the destination directory
            file(MAKE_DIRECTORY "${destination_dir}/${file}")
        else()
            # Copy the file
            file(COPY "${file}" DESTINATION "${destination_dir}")
        endif()
    endforeach()
endfunction()