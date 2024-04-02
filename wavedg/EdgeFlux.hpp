#ifndef DG_EDGE_FLUX_HPP
#define DG_EDGE_FLUX_HPP

#include <limits>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
#include "lagrange_interpolation.hpp"
#include "linalg.hpp"
#include "Operator.hpp"

namespace dg
{
    /// @brief computes the numerical flux: \f$a \langle \{C u\}, [v] \rangle + b \langle |C| [u], [v] \rangle\f$.
    ///
    /// @details For the DG discretization:
    /// \f[(\mathbf{A} u, \nabla v) + a \langle \{C u\}, [v] \rangle + b \langle [ |C| u ], [v] \rangle\f]
    ///
    /// We have:
    ///
    /// `Div` + `EdgeFlux`
    ///
    /// Where `EdgeFlux` computes the trace integrals o \f$\partial I\f$. The
    /// integral can be computed using the quadrature rule of the basis
    /// function, or by supplying a higher order quadrature rule. If using the
    /// quadrature rule of the basis, no projections are computed and thus there
    /// may be aliasing errors when \f$A\f$ is not constant, nevertheless, the
    /// computation is faster. When the quadrature rule is the same as the basis
    /// with \f$p\f$ points, then the cost is \f$O(n p)\f$, if the quadrature
    /// rule is different and has \f$q\f$ points then the cost is \f$O(n p q)\f$.
    ///
    /// The default values a = -1 and b = -1/2 correspond to the upwind flux.
    template <bool ApproxQuadrature>
    class EdgeFlux : public Operator
    {
    public:
        /// @brief constructs the EdgeFlux
        /// @param[in] nvar vector dimension of u
        /// @param[in] mesh
        /// @param[in] edge_type the edges to apply the fluxes to
        /// @param[in] basis collocation points for basis functions
        /// @param[in] A the prolongation of the coefficient A on the edge
        /// faces. if @a constant_coefficient, shape (n_var, n_var, 2),
        /// otherwise shape (2, nvar, nvar, 2, n_colloc, n_edges). See FaceProlongator.
        /// @param[in] constant_coefficient specify if A is constant throughout the mesh.
        /// @param[in] a 
        /// @param[in] b 
        /// @param[in] quad quadrature rule for computing integrals. If @a ApproxQuadrature == true, then quad is not referenced.
        EdgeFlux(int nvar, const Mesh2D& mesh, FaceType edge_type, const QuadratureRule * basis, const double * A, bool constant_coefficient, double a=-1.0, double b=-0.5, const QuadratureRule * quad = nullptr);

        EdgeFlux(int nvar, const Mesh1D& mesh, FaceType edge_type, const QuadratureRule * basis, const double * A, bool constant_coefficient, double a=-1.0, double b=-0.5, const QuadratureRule * quad = nullptr);

        ~EdgeFlux() = default;

        /// @brief applies the trace integral: \f$a \langle {C u}, [v] \rangle + b \langle |C| [u], [v] \rangle\f$.
        /// @param[in] uB the values of u on the edges. Has shape `(2, n_var, n_colloc, n_elem)`.
        /// @param[in,out] Fb on exit, the flux on the first edge, and the
        /// negative flux on the second edge. Has shape `(2, n_var, n_colloc, n_elem)`. It is safe to set Fb = uB to apply the flux inplace.
        void action(const double * uB, double * Fb) const override;

        /// @brief applies the trace integral: \f$a \langle {C u}, [v] \rangle + b \langle |C| [u], [v] \rangle\f$.
        /// @param[in] n_var IGNORED
        /// @param[in] uB the values of u on the edges. Has shape `(2, n_var, n_colloc, n_elem)`.
        /// @param[in,out] Fb on exit, the flux on the first edge, and the
        /// negative flux on the second edge. Has shape `(2, n_var, n_colloc, n_elem)`. It is safe to set Fb = uB to apply the flux inplace.
        void action(int n_var, const double * uB, double * Fb) const override
        {
            action(uB, Fb);
        }

    private:
        const FaceType etype;
        const int dim;
        const int n_edges;
        const int n_colloc;
        const int n_var;

        int n_quad;

        dmat P; // (n_quad, n_colloc)
        dmat Pt; // transpose of P
        dvec F;

        mutable dvec uf; // (2, n_var)
        mutable dvec Uq; // (n_var, n_quad)

    };
} // namespace dg


#endif