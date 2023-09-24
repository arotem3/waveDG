#ifndef WAVE_DG_HPP
#define WAVE_DG_HPP

#include "config.hpp" // MUST RUN CMAKE FIRST!

#ifdef WDG_USE_MPI
#include <mpi.h>
#endif

#include "wavedg/BasisProduct.hpp"
#include "wavedg/Div.hpp"
#include "wavedg/Edge.hpp"
#include "wavedg/EdgeFlux.hpp"
#include "wavedg/eig.hpp"
#include "wavedg/Element.hpp"
#include "wavedg/FaceProjector.hpp"
#include "wavedg/jacobi.hpp"
#include "wavedg/lagrange_interpolation.hpp"
#include "wavedg/MassMatrix.hpp"
#include "wavedg/Mesh2D.hpp"
#include "wavedg/ode.hpp"
#include "wavedg/Projector.hpp"
#include "wavedg/QuadratureRule.hpp"
#include "wavedg/Tensor.hpp"
#include "wavedg/WaveEquation.hpp"

#endif