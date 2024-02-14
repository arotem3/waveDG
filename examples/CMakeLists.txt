##### --- examples --- #####

##### advection #####
add_executable(advection advection.cpp)
target_link_libraries(advection PUBLIC wavedg)
add_custom_command(TARGET advection POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### convergence2d #####
add_executable(convergence2d convergence2d.cpp)
target_link_libraries(convergence2d PUBLIC wavedg)
add_custom_command(TARGET convergence2d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec1d #####
add_executable(advec1d advec1d.cpp)
target_link_libraries(advec1d PUBLIC wavedg)
add_custom_command(TARGET advec1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec-mpi #####
add_executable(advec-mpi advec-mpi.cpp)
target_link_libraries(advec-mpi PUBLIC wavedg)
add_custom_command(TARGET advec-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
