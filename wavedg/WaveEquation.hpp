#ifndef DG_WAVE_EQUATION_HPP
#define DG_WAVE_EQUATION_HPP

#include "config.hpp"
#include "Div.hpp"
#include "FaceProjector.hpp"
#include "EdgeFlux.hpp"
#include "Projector.hpp"
#include "MassMatrix.hpp"

namespace dg
{
    // DG discretization of the operator:
    // d/dx A(i, j, 0) * u(j) + d/dy A(i, j, 1) * u(j)
    // --> - ( A(i, j, 0) u(j), d/dx v ) - ( A(i, j, 1) u(j), d/dy v )
    //          + a < (n.A)(i, j) {u(j)}, [v] >
    //          + b < |n.A|(i, j) [u(j)], [v] >
    class Advection
    {
    private:
        std::unique_ptr<Div> div;
        std::unique_ptr<EdgeFlux> FlxB;
        std::unique_ptr<EdgeFlux> FlxI;
        FaceProjector ProjB;
        FaceProjector ProjI;

        mutable dcube uB;
        mutable dcube uI;

    public:
        Advection(const Mesh2D& mesh, const QuadratureRule * basis);

        /// @brief Apply the advection operation
        /// @param x shape (n_colloc, n_colloc, n_elem)
        /// @param y shape (n_colloc, n_colloc, n_elem)
        void operator()(const double * x, double * y) const;
    };

    // class WaveEquation
    // {
    // private:
    //     std::unique_ptr<Div> div;
    //     std::unique_ptr<EdgeFlux> FlxB;
    //     std::unique_ptr<EdgeFlux> FlxI;
    //     FaceProjector ProjB;
    //     FaceProjector ProjI;

    //     mutable Tensor<4, double> uB;
    //     mutable Tensor<4, double> uI;
    // };
} // namespace dg

#endif