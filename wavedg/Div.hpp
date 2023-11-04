#ifndef DG_DIV_HPP
#define DG_DIV_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "Operator.hpp"

namespace dg
{
    /// @brief Computes the bilinear form \f$(\mathbf{A} u, grad v)_I\f$.
    ///
    /// @details For the DG discretization:
    /// $$(\mathbf{A} u, grad v)_I - a \lbracket {C u}, [v] \rbracket_{\partial I} - b \lbracket |C| [u], [v] \rbracket_{\partial I}$$
    /// We have:
    /// `Div` - `EdgeFlux`
    /// where `Div` computes the volume integrals on \f$I\f$. The integral can be
    /// computed using the quadrature rule of the basis function, or by
    /// supplying a higher order quadrature rule. If using the quadrature rule
    /// of the basis, no projections are used and thus there may be aliasing
    /// errors when \f$A\f$ is not constant, but the computation is faster.
    template <bool ApproxQuadrature>
    class Div : public Operator
    {
    private:
        const int n_var;
        const int n_colloc;
        const int n_elem;

        int n_quad;

        dmat D; // (n_quad, n_colloc)
        dmat Dt; // transpose of D
        dmat P; //(n_quad, n_colloc)
        dmat Pt; // transpose of P
        dvec _op; // (2, n_var, n_var, n_quad, n_quad, n_elem): mapping to the flux on every element
        
        mutable dcube Uq;
        mutable Tensor<4, double> Fq; // (2, n_var, n_quad, n_quad)
        mutable dcube Df;
        mutable dcube Pg;
        mutable dcube_wrapper Pu;

    public:
        /// @brief initialize a Div object for the bilinear form: \f$-(A^{0} u, v_x) - (A^{1} u, v_y)\f$.
        /// @param nvar vector dimension of grid function.
        /// @param mesh mesh.
        /// @param basis collocation point for Lagrange basis.
        /// @param A coefficient if constant coefficient then shape is `(n_var, n_var, 2)`,
        /// else `(n_var, n_var, 2, n_colloc, n_colloc, n_elem)`. Where
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(d, k, \ell, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule for computing integrals. if (ApproxQuadrature) then quad is not referenced.
        Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        /// @brief computes du <- du + (A u, grad v)
        /// @param u 
        /// @param du 
        void action(const double * u, double * du) const override;
    };
} // namespace dg


#endif