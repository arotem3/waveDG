add_executable(advection advection.cpp)
target_link_libraries(advection PUBLIC wavedg)
# if(WDG_USE_MPI)
# 	message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} ./examples/advection ${MPIEXEC_POSTFLAGS}")
# endif()

add_executable(distribute_mesh distribute_mesh.cpp)
target_link_libraries(distribute_mesh PUBLIC wavedg)

add_executable(advec-mpi advec-mpi.cpp)
target_link_libraries(advec-mpi PUBLIC wavedg)