##### --- examples --- #####

##### advec1d-mpi #####
add_executable(advec1d-mpi advec1d-mpi.cpp)
target_link_libraries(advec1d-mpi PUBLIC wavedg)
add_custom_command(TARGET advec1d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave2d-mpi #####
add_executable(wave2d-mpi wave2d-mpi.cpp)
target_link_libraries(wave2d-mpi PUBLIC wavedg)
add_custom_command(TARGET wave2d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz2d #####
add_executable(waveholtz2d waveholtz2d.cpp)
target_link_libraries(waveholtz2d PUBLIC wavedg)
add_custom_command(TARGET waveholtz2d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave2d #####
add_executable(wave2d wave2d.cpp)
target_link_libraries(wave2d PUBLIC wavedg)
add_custom_command(TARGET wave2d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### cl1d #####
add_executable(cl1d cl1d.cpp)
target_link_libraries(cl1d PUBLIC wavedg)
add_custom_command(TARGET cl1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec2d #####
add_executable(advec2d advec2d.cpp)
target_link_libraries(advec2d PUBLIC wavedg)
add_custom_command(TARGET advec2d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz1d-mpi #####
add_executable(waveholtz1d-mpi waveholtz1d-mpi.cpp)
target_link_libraries(waveholtz1d-mpi PUBLIC wavedg)
add_custom_command(TARGET waveholtz1d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec2d-mpi #####
add_executable(advec2d-mpi advec2d-mpi.cpp)
target_link_libraries(advec2d-mpi PUBLIC wavedg)
add_custom_command(TARGET advec2d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave1d-mpi #####
add_executable(wave1d-mpi wave1d-mpi.cpp)
target_link_libraries(wave1d-mpi PUBLIC wavedg)
add_custom_command(TARGET wave1d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### advec1d #####
add_executable(advec1d advec1d.cpp)
target_link_libraries(advec1d PUBLIC wavedg)
add_custom_command(TARGET advec1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### wave1d #####
add_executable(wave1d wave1d.cpp)
target_link_libraries(wave1d PUBLIC wavedg)
add_custom_command(TARGET wave1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz2d-mpi #####
add_executable(waveholtz2d-mpi waveholtz2d-mpi.cpp)
target_link_libraries(waveholtz2d-mpi PUBLIC wavedg)
add_custom_command(TARGET waveholtz2d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### waveholtz1d #####
add_executable(waveholtz1d waveholtz1d.cpp)
target_link_libraries(waveholtz1d PUBLIC wavedg)
add_custom_command(TARGET waveholtz1d POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)

##### cl1d-mpi #####
add_executable(cl1d-mpi cl1d-mpi.cpp)
target_link_libraries(cl1d-mpi PUBLIC wavedg)
add_custom_command(TARGET cl1d-mpi POST_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ../solution)
