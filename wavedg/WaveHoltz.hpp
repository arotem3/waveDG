#ifndef DG_WAVEHOLTZ_HPP
#define DG_WAVEHOLTZ_HPP

#include "wdg_config.hpp"
#include "WaveEquation.hpp"
#include "Mesh2D.hpp"
#include "MassMatrix.hpp"
#include "ode.hpp"

namespace dg
{
    class WaveHoltz
    {
    public:
        WaveHoltz(double omega, const Mesh2D& mesh, const QuadratureRule * basis, const int * boundary_conditions, bool approx_quad=false);

        inline double K(double t) const
        {
            return omega / M_PI * (std::cos(omega * t) - 0.25);
        }

        /// @brief computes \f$\Pi(0) = \int_0^T K(t) U(t) \, dt\f$ where \f$U\f$ is
        /// the solution to the wave equation with forcing \f$F \sin(\omega t)\f$
        /// and zero initial conditions.
        void pi0(double * U, const double * F) const;

        /// @brief Computes \f$S u = \Pi(u) - \Pi(0) = \int_0^T K(t) U(t) \, dt\f$
        /// where \f$U\f$ is the solution to homogeneous wave equation with initial
        /// conditions \f$u\f$
        void S(double * U) const;

        /// @brief computes the solution to the complex valued Helmholtz problem
        /// found by WaveHoltz iteration
        /// @param H is overwritten with the solution to the Helmholtz problem. Has
        /// dimension (2, n) where n is the number of collocation points, the values
        /// are stored as (real, imag) pairs exactly like complex<double>.
        /// @param U the fixed point of the WaveHoltz iteration, i.e. solves S*U =
        /// pi0. Has dimension (3, n)
        void postprocess(double * H, const double * U) const;

    private:
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