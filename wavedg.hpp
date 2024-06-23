#ifndef WAVE_DG_HPP
#define WAVE_DG_HPP

#include "wdg_config.hpp" // MUST RUN CMAKE FIRST!

#include "wavedg/Tensor.hpp"
#include "wavedg/linalg.hpp"

#include "FEMVector.hpp"

#include "wavedg/jacobi.hpp"
#include "wavedg/lagrange_interpolation.hpp"
#include "wavedg/QuadratureRule.hpp"

#include "wavedg/Element.hpp"
#include "wavedg/Edge.hpp"
#include "wavedg/Mesh2D.hpp"

#include "wavedg/MassMatrix.hpp"
#include "wavedg/LinearFunctional.hpp"
#include "wavedg/FaceProlongator.hpp"

#include "wavedg/DivF.hpp"
#include "wavedg/EdgeFluxF.hpp"
#include "wavedg/Div.hpp"
#include "wavedg/EdgeFlux.hpp"

#include "wavedg/Nabla.hpp"

#include "wavedg/StiffnessMatrix.hpp"
#include "wavedg/DG2CG.hpp"
#include "wavedg/ZeroBoundary.hpp"
#include "wavedg/CGMask.hpp"

#include "wavedg/pcg.hpp"

#include "wavedg/ode.hpp"

#include "wavedg/Advection.hpp"
#include "wavedg/WaveEquation.hpp"
#include "wavedg/WaveHoltz.hpp"

#endif