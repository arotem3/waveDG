cmake_minimum_required(VERSION 3.10)

project(wavedg)

configure_file(config.hpp.in wavedg/wdg_config.hpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# LAPACK
find_package(LAPACK REQUIRED)

# specify which version of waveDG to make
option(WDG_DEBUG "compiles library in debug mode and add extra checks for correctness (like index bounds, etc.)")
option(WDG_USE_MPI "replace serial implementation with MPI parallel implementation. (CANNOT be combined with WDG_USE_OMP)")
#option(WDG_USE_OMP "replace serial implementation with OpenMP parallel implementation. (CANNOT be combied with WDG_USE_MPI)")

add_compile_options(-Wall)
if (WDG_DEBUG)
    add_compile_options(-g)
else()
    add_compile_options(-O3) # if not debug then optimize
endif()

if(WDG_USE_MPI AND WDG_USE_OMP)
    message(FATAL_ERROR "Mixed MPI and OpenMP not supported.")
endif()

if(WDG_USE_MPI)
    find_package(MPI REQUIRED)
endif()

# if (WDG_USE_OMP)
#     find_package(OpenMP REQUIRED)
# endif()

add_library(
    wavedg SHARED
    
    wavedg.hpp
    
    source/Div.cpp
    source/Edge.cpp
    source/EdgeFlux.cpp
    source/eig.cpp
    source/Element.cpp
    source/FaceProlongator.cpp
    source/jacobi.cpp
    source/lagrange_interpolation.cpp
    source/linalg.cpp
    source/MassMatrix.cpp
    source/Mesh2D.cpp
    source/QuadratureRule.cpp
    source/FaceProjector.cpp
    source/WaveEquation.cpp
    source/jacobi.cpp
)

target_include_directories(wavedg PUBLIC . ./wavedg)

target_link_libraries(wavedg PUBLIC ${LAPACK_LIBRARIES})

if (WDG_USE_MPI)
    target_link_libraries(wavedg PUBLIC MPI::MPI_CXX)
endif()

# if (WDG_USE_OMP)
#     target_link_libraries(wavedg PUBLIC OpenMP::OpenMP_CXX)
# endif()

set_target_properties(wavedg PROPERTIES PUBLIC_HEADER wavedg.hpp)

install(TARGETS wavedg
            LIBRARY DESTINATION lib
            PUBLIC_HEADER DESTINATION include
)
install(
    DIRECTORY wavedg DESTINATION include
)

add_subdirectory(examples)
add_subdirectory(tests)