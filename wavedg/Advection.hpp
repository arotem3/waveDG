#ifndef WDG_ADVECTION_HPP
#define WDG_ADVECTION_HPP

#include "wdg_config.hpp"
#include "Div.hpp"
#include "FaceProlongator.hpp"
#include "EdgeFlux.hpp"
#include "Projector.hpp"
#include "MassMatrix.hpp"

namespace dg
{
    /// DG discretization of the operator:
    /// -d/dx A(i, j, 0) * u(j) - d/dy A(i, j, 1) * u(j)
    /// --> ( A(i, j, 0) u(j), d/dx v ) + ( A(i, j, 1) u(j), d/dy v )
    ///          - a < (n.A)(i, j) {u(j)}, [v] >
    ///          - b < |n.A|(i, j) [u(j)], [v] >
    template <bool ApproxQuadrature>
    class Advection : public Operator
    {
    private:
        const int n_var;

        std::unique_ptr<Operator> div;
        std::unique_ptr<Operator> Flx;
        std::unique_ptr<FaceProlongator> face_prol;

        mutable dvec uI;

    public:
        /// @brief initializes advection operator: \f$-(A^{0} u, v_x)_I - (A^{1} u, v_y)_I + a \lbracket {C u}, [v] \rbracket_{\partial I} + b \lbracket |C| [u], [v] \rbracket_{\partial I}\f$
        /// @param n_var vector dimension of u
        /// @param mesh mesh
        /// @param basis collocation points for Lagrange basis
        /// @param a coefficient A. If constant coefficient then shape is `(n_var, n_var, 2)`
        /// else `(n_var, n_var, 2, n_colloc, n_colloc, n_elem)`. Where
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(d, k, \ell, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule. If ApproxQuadrature, then @a quad is not referenced.
        Advection(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        /// @brief Apply the advection operation
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem)
        /// @param divF shape (n_var, n_colloc, n_colloc, n_elem)
        void action(const double * u, double * divF) const override;
    };

    template <bool ApproxQuad>
    Advection<ApproxQuad>::Advection(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad)
        : n_var(n_var_)
    {
        const int neI = mesh.n_edges(Edge::INTERIOR);
        const int n_colloc = basis->n;

        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(mesh, basis, Edge::INTERIOR);

        dvec aI;
        if (constant_coefficient)
        {
            aI.reshape(2 * v2d);
            for (int i=0; i < 2*v2d; ++i)
                aI[i] = a[i];
        }
        else
        {
            aI.reshape(4 * v2d * n_colloc * neI); // A prolonged to interior edges
            face_prol->action(a, aI, 2*v2d);
        }
        
        div.reset(new Div<ApproxQuad>(n_var, mesh, basis, a, constant_coefficient, quad));

        // take negative a,b since we are computing -div rather than div.
        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, Edge::INTERIOR, basis, aI, constant_coefficient, -1.0, -0.5, quad));
        uI.reshape(2 * n_var * n_colloc * neI);
    }

    template <bool ApproxQuad>
    void Advection<ApproxQuad>::action(const double * u, double * divF) const
    {
        // volume integral
        div->action(u, divF);

        face_prol->action(u, uI, n_var);
        Flx->action(uI, uI);
        face_prol->t(uI, divF, n_var);
    }

    /// @brief Specifies that u == 0 outside domain.
    /// @tparam ApproxQuadrature whether to use approximate quadrature for face integrals.
    template <bool ApproxQuadrature>
    class AdvectionHomogeneousBC : public Operator
    {
    private:
        const int n_var;

        std::unique_ptr<FaceProlongator> face_prol;
        std::unique_ptr<Operator> Flx;

        mutable dvec uB;

    public:
        /// @brief initializes boundary conditions operator
        /// @param n_var vector dimension of u
        /// @param mesh mesh
        /// @param basis collocation points for Lagrange basis
        /// @param a coefficient A. If constant coefficient then shape is `(n_var, n_var, 2)`
        /// else `(n_var, n_var, 2, n_colloc, n_colloc, n_elem)`. Where
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(d, k, \ell, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule. If ApproxQuadrature, then @a quad is not referenced.
        AdvectionHomogeneousBC(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        /// @brief adds boundary conditions
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem)
        /// @param divF shape (n_var, n_colloc, n_colloc, n_elem). On exit, divF <- divF + BC
        void action(const double * u, double * divF) const override;
    };

    template <bool ApproxQuad>
    AdvectionHomogeneousBC<ApproxQuad>::AdvectionHomogeneousBC(int nv, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad)
        : n_var(nv)
    {
        const int neB = mesh.n_edges(Edge::BOUNDARY);
        const int n_colloc = basis->n;
        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(mesh, basis, Edge::BOUNDARY);

        dvec aB;
        if (constant_coefficient)
        {
            aB.reshape(2 * v2d);
            for (int i=0; i < 2*v2d; ++i)
                aB[i] = a[i];
        }
        else
        {
            aB.reshape(4 * v2d * n_colloc * neB); // A prolonged to boundary edges
            face_prol->action(a, aB, 2*v2d);
        }

        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, Edge::BOUNDARY, basis, aB, constant_coefficient, -1.0, -0.5, quad));
        uB.reshape(2 * n_var * n_colloc * neB);
    }

    template <bool ApproxQuad>
    void AdvectionHomogeneousBC<ApproxQuad>::action(const double * u, double * divF) const
    {
        face_prol->action(u, uB, n_var);
        Flx->action(uB, uB);
        face_prol->t(uB, divF, n_var);
    }
} // namespace dg

#endif