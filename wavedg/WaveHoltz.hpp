#ifndef DG_WAVEHOLTZ_HPP
#define DG_WAVEHOLTZ_HPP

#include "wdg_config.hpp"
#include "WaveEquation.hpp"
#include "Mesh2D.hpp"
#include "MassMatrix.hpp"
#include "ode.hpp"

namespace dg
{
    /// @brief Implements the WaveHoltz operators used to compute the solution to the Helmholtz equation:
    ///         $$\Delta H + \omega^2 H = f.$$
    /// By solving the closely relate wave equation:
    ///     $$p_t + \nabla\cdot \vec{u} = F(x,t),$$
    ///     $$\vec{u}_t + \nabla p = 0.$$
    ///
    /// @details The WaveHoltz algorithm solves the Helmholtz equation via the fixed point iteration:
    ///         $$w^{(n+1)} = S w^{(n)} + \pi_0.$$
    /// Where \f$S w^{(n)}\f$ is computed by solving the wave equation with \f$F\equiv 0\f$ and  initial condition \f$w^{(n)}\f$,
    /// and $\pi_0$ is similarly computed by solving the wave equation with forcing \f$F(x,t) = -f(x)\sin(\omega t)/\omega\f$.
    /// Here \f$w = (p, \vec{u})\f$.
    ///
    /// Once a fixed point \f$w^{(*)}\f$ is found, the complex valued solution to the original Helmholtz problem
    /// is \f$H = p + \frac{1}{i\omega}\nabla\cdot \vec{u}\f$ which can be computed with the `postprocess` function.
    class WaveHoltz
    {
    public:
        /// @brief Initializes WaveHoltz operators
        /// @param omega Helmholtz frequency
        /// @param mesh 2d mesh
        /// @param basis collocation/quadrature points for basis functions
        /// @param boundary_conditions Specifies the boundary condition on each boundary edge as either absorbing (0) or reflecting/Neumann (1).
        /// @param approx_quad whether to compute DG operators with exact or approximate quadrature (faster)
        WaveHoltz(double omega, const Mesh2D& mesh, const QuadratureRule * basis, const int * boundary_conditions, bool approx_quad=false);

        /// @brief Initializes WaveHoltz operators
        /// @param omega Helmholtz frequency
        /// @param mesh 2d mesh
        /// @param basis collocation/quadrature points for basis functions
        /// @param boundary_conditions Specifies the boundary condition on each boundary edge as either absorbing (0) or reflecting/Neumann (1).
        /// @param approx_quad whether to compute DG operators with exact or approximate quadrature (faster)
        WaveHoltz(double omega, const Mesh1D& mesh, const QuadratureRule * basis, const int * boundary_conditions, bool approx_quad=false);

        /// @brief WaveHoltz kernel.
        /// @param t 
        /// @return 2/T (cos(omega t) - 1/4)
        inline double K(double t) const
        {
            return omega / M_PI * (std::cos(omega * t) - 0.25);
        }

        /// @brief computes \f$\Pi(0) = \int_0^T K(t) w(t) \, dt\f$ where \f$w\f$ is
        /// the solution to the wave equation with forcing \f$-F \sin(\omega t)/\omega\f$
        /// and zero initial conditions.
        void pi0(double * w, const double * F) const;

        /// @brief Computes \f$S w = \Pi(w) - \Pi(0) = \int_0^T K(t) W(t) \, dt\f$
        /// where \f$W\f$ is the solution to homogeneous wave equation with initial
        /// conditions \f$w\f$
        void S(double * w) const;

        /// @brief computes the solution to the complex valued Helmholtz problem
        /// found by WaveHoltz iteration
        /// @param H is overwritten with the solution to the Helmholtz problem. Has
        /// dimension (2, n) where n is the number of collocation points, the values
        /// are stored as (real, imag) pairs exactly like complex<double>.
        /// @param w the fixed point of the WaveHoltz iteration, i.e. solves w = S*w
        /// pi0. Has dimension (3, n)
        void postprocess(double * H, const double * w) const;

    private:
        const int dim;
        const double omega;
        const int ndof;
        
        double T;
        double dt;
        int nt;
        
        std::unique_ptr<InvertibleOperator> m;
        std::unique_ptr<Operator> a;
        std::unique_ptr<Operator> bc;
        
        ode::RungeKutta4 rk;

        mutable dvec p;
    };
} // namespace dg

#endif