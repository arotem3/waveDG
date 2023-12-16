##### --- examples --- #####

##### distribute_mesh #####
add_executable(distribute_mesh distribute_mesh.cpp)
target_link_libraries(distribute_mesh PUBLIC wavedg)
add_custom_command(TARGET distribute_mesh POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advection #####
add_executable(advection advection.cpp)
target_link_libraries(advection PUBLIC wavedg)
add_custom_command(TARGET advection POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec-mpi #####
add_executable(advec-mpi advec-mpi.cpp)
target_link_libraries(advec-mpi PUBLIC wavedg)
add_custom_command(TARGET advec-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
