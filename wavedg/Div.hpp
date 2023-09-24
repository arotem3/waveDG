#ifndef DG_DIV_HPP
#define DG_DIV_HPP

#include "config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    // volume integral for the operator
    // d/dx A(i, j, 0) * u(j) + d/dy A(i, j, 1) * u(j)
    // --> - ( A(i, j, 0) u(j), d/dx v ) - ( A(i, j, 1) u(j), d/dy v )
    //          + a < (n.A)(i, j) {u(j)}, [v] >
    //          + b < |n.A|(i, j) [u(j)], [v] >
    // --> Div<A>(u) + EdgeFlux<A, a, b>(u)
    // The integral can be computed using the quadrature rule of the basis
    // function, or by suplying a higher order quadrature rule.
    class Div
    {
    private:
        const int n_var;
        const int n_colloc;
        const int n_quad;
        const int n_elem;

        const bool approx_quadrature;

        const QuadratureRule * quad;

        dmat D;
        dmat P;
        std::vector<double> _op; // (n_var, n_var, 2, n_quad, n_quad, n_elem)
        
        mutable dcube Uq; // (n_var, n_quad, n_quad)
        mutable Tensor<4, double> Fq; // (n_var, n_quad, n_quad, 2)
        mutable dcube PF; // (n_var, n_colloc, n_quad)

    public:
        Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient);
        Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad, const double * A, bool constant_coefficient);

        void operator()(const double * u, double * du) const;
    };
} // namespace dg


#endif