cmake_minimum_required(VERSION 3.10)

project(WaveDG)

# Find all .cpp files in the current directory
file(GLOB CPP_FILES "*.cpp")

# Add an executable for each .cpp file and link it against the wavedg library
foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(EXECUTABLE_NAME ${CPP_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${CPP_FILE})
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE wavedg)
    add_custom_command(TARGET ${EXECUTABLE_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/solution)
endforeach()