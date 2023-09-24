#ifndef DG_EDGE_FLUX_HPP
#define DG_EDGE_FLUX_HPP

#include <limits>

#include "config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "eig.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    // numerical flux for the operator:
    // d/dx A(i, j, 0) * u(j) + d/dy A(i, j, 1) * u(j)
    // --> - ( A(i, j, 0) u(j), d/dx v ) - ( A(i, j, 1) u(j), d/dy v )
    //          + a < (n.A)(i, j) {u(j)}, [v] >
    //          + b < |n.A|(i, j) [u(j)], [v] >
    // --> Div<A>(u) + EdgeFlux<A, a, b>(u)
    // Here (n.A)(i, j) = n(0)*A(i,j,0) + n(1)*A(i,j,1), n.A == R*D*inv(R) -> |n.A| = R*|D|*inv(R).
    // the default values a = 1 and b = -1/2 is the upwind flux
    // The integral can be computed using the quadrature rule of the basis
    // function, or by suplying a higher order quadrature rule.
    class EdgeFlux
    {
    private:
        const EdgeType etype;
        const int n_edges;
        const int n_quad;
        const int n_colloc;
        const int n_var;
        const bool use_colloc;

        dmat P; // (n_quad, n_colloc)
        Tensor<5,double> F; // (n_var, n_var, 2, n_quad, n_edges)
        const double * _ds; // (n_quad, n_edges)
        const QuadratureRule * quad;

        mutable dmat uf; // (n_var, 2)
        mutable dcube Uq; // (n_var, n_quad, 2)

    public:
        /// @brief constructs the EdgeFlux
        /// @param nvar vector dimension of u
        /// @param mesh
        /// @param edge_type the edges to apply the fluxes to
        /// @param basis collocation points for basis functions. Integrals are
        /// computed using this quadrature rule.
        /// @param nA shape (nvar, nvar, n_colloc, n_edges) the (projection of
        /// the) coefficient A dotted with n, that is n.A (see
        /// EdgeProjector::project_normal_coefficient)
        /// @param a 
        /// @param b 
        EdgeFlux(int nvar, const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule * basis, const double * nA, double a=1.0, double b=-0.5);
        
        /// @brief constructs the EdgeFlux
        /// @param nvar vector dimension of u
        /// @param mesh
        /// @param edge_type the edges to apply the fluxes to
        /// @param basis collocation points for basis functions
        /// @param quad quadrature rule for computing integrals
        /// @param nA shape (nvar, nvar, n_colloc, n_edges) the (projection of
        /// the) coefficient A dotted with n, that is n.A (see
        /// EdgeProjector::project_normal_coefficient)
        /// @param a 
        /// @param b 
        EdgeFlux(int nvar, const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule * basis, const QuadratureRule * quad, const double * nA, double a=1.0, double b=-0.5);

        void operator()(double *) const;
    };
} // namespace dg


#endif