#ifndef WAVE_DG_HPP
#define WAVE_DG_HPP

#include "wdg_config.hpp" // MUST RUN CMAKE FIRST!

#include "wavedg/Tensor.hpp"
#include "wavedg/linalg.hpp"

#include "wavedg/jacobi.hpp"
#include "wavedg/lagrange_interpolation.hpp"
#include "wavedg/QuadratureRule.hpp"

#include "wavedg/Element.hpp"
#include "wavedg/Edge.hpp"
#include "wavedg/Mesh2D.hpp"

#include "wavedg/MassMatrix.hpp"
#include "wavedg/Projector.hpp"
#include "wavedg/LinearFunctional.hpp"
#include "wavedg/FaceProlongator.hpp"

#include "wavedg/Div.hpp"
#include "wavedg/EdgeFlux.hpp"

#include "wavedg/ode.hpp"

#include "wavedg/Advection.hpp"

#endif