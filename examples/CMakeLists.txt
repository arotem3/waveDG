##### --- examples --- #####

##### advec1d-mpi #####
add_executable(advec1d-mpi advec1d-mpi.cpp)
target_link_libraries(advec1d-mpi PUBLIC wavedg)
add_custom_command(TARGET advec1d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advection #####
add_executable(advection advection.cpp)
target_link_libraries(advection PUBLIC wavedg)
add_custom_command(TARGET advection POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz #####
add_executable(waveholtz waveholtz.cpp)
target_link_libraries(waveholtz PUBLIC wavedg)
add_custom_command(TARGET waveholtz POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave #####
add_executable(wave wave.cpp)
target_link_libraries(wave PUBLIC wavedg)
add_custom_command(TARGET wave POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec1d #####
add_executable(advec1d advec1d.cpp)
target_link_libraries(advec1d PUBLIC wavedg)
add_custom_command(TARGET advec1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec-mpi #####
add_executable(advec-mpi advec-mpi.cpp)
target_link_libraries(advec-mpi PUBLIC wavedg)
add_custom_command(TARGET advec-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz-mpi #####
add_executable(waveholtz-mpi waveholtz-mpi.cpp)
target_link_libraries(waveholtz-mpi PUBLIC wavedg)
add_custom_command(TARGET waveholtz-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave-mpi #####
add_executable(wave-mpi wave-mpi.cpp)
target_link_libraries(wave-mpi PUBLIC wavedg)
add_custom_command(TARGET wave-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec #####
add_executable(advec advec.cpp)
target_link_libraries(advec PUBLIC wavedg)
add_custom_command(TARGET advec POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
