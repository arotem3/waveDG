#ifndef DG_DIV_HPP
#define DG_DIV_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
#include "lagrange_interpolation.hpp"
#include "Operator.hpp"

namespace dg
{
    /// @brief Computes the bilinear form \f$(\mathbf{A} u, \nabla v)_I\f$.
    ///
    /// @details For the DG discretization:
    /// \f[(\mathbf{A} u, \nabla v) + a \langle \{C u\}, [v] \rangle + b \langle [ |C| u ], [v] \rangle \f]
    /// We have:
    /// `Div` + `EdgeFlux`
    /// where `Div` computes the volume integrals on \f$I\f$. The integral can be
    /// computed using the quadrature rule of the basis function, or by
    /// supplying a higher order quadrature rule. If using the quadrature rule
    /// of the basis, no projections are used and thus there may be aliasing
    /// errors when \f$A\f$ is not constant, but the computation is faster.
    template <bool ApproxQuadrature>
    class Div : public Operator
    {
    public:
        /// @brief initialize a Div object for the bilinear form: \f$(\mathbf{A}u, \nabla v) = (A^{0} u, v_x) + (A^{1} u, v_y)\f$.
        /// @param nvar vector dimension of grid function.
        /// @param mesh 2D mesh.
        /// @param basis collocation point for Lagrange basis.
        /// @param A coefficient if constant coefficient then shape is `(n_var, n_var, 2)`,
        /// else `(n_var, n_var, 2, n_colloc, n_colloc, n_elem)`. Where
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(k, \ell, d, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule for computing integrals. if (ApproxQuadrature) then quad is not referenced.
        Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        /// @brief initialize a Div object for the bilinear form: \f$(A u, v_x)\f$.
        /// @param nvar vector dimension of grid function.
        /// @param mesh 1D mesh.
        /// @param basis collocation point for Lagrange basis.
        /// @param A coefficient if constant coefficient then shape is `(n_var, n_var)`,
        /// else `(n_var, n_var, n_colloc, n_elem)`. Where on element el and collocation
        /// point i: \f$A_{k,\ell}=A(k, \ell, i, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule for computing integrals. if (ApproxQuadrature) then quad is not referenced.
        Div(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        ~Div() = default;

        /// @brief computes du <- du + (A u, grad v)
        /// @param u 
        /// @param du 
        void action(const double * u, double * du) const override;
    
    private:
        const int dim;
        const int n_var;
        const int n_colloc;
        const int n_elem;

        int n_quad;

        dmat D; // (n_quad, n_colloc)
        dmat Dt; // transpose of D
        dmat P; //(n_quad, n_colloc)
        dmat Pt; // transpose of P
        dvec _op; // mapping to the flux on every element
        
        mutable dvec Uq;
        mutable dvec Fq;
        mutable dvec Df;
        mutable dvec Pg;
    };
} // namespace dg


#endif