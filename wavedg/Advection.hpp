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
    /// @brief DG discretization of the operator:
    /// \f[-\nabla \cdot \mathbf{A} u \longrightarrow (\mathbf{A} u, \nabla v) + a \langle \{ C u \} , [v] \rangle + b \langle [|C|u], [v] \rangle\f]
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
        /// @brief initializes advection operator: \f$(\mathbf{A} u, \nabla v) + a \langle \{ C u \} , [v] \rangle + b \langle [|C|u], [v] \rangle\f$
        /// @param n_var vector dimension of u
        /// @param mesh mesh
        /// @param basis collocation points for Lagrange basis
        /// @param a coefficient A. If constant coefficient then shape is `(n_var, n_var, 2)`
        /// else `(n_var, n_var, 2, n_colloc, n_colloc, n_elem)`. Where
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(k, \ell, d, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule. If ApproxQuadrature, then @a quad is not referenced.
        Advection(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        ~Advection() = default;

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

        face_prol = make_face_prolongator(n_var, mesh, basis, Edge::INTERIOR);

        dvec aI;
        if (constant_coefficient)
        {
            aI.reshape(2 * v2d);

            for (int i=0; i < 2*v2d; ++i)
                aI[i] = a[i];
        }
        else
        {
            auto f = make_face_prolongator(2*v2d, mesh, basis, Edge::INTERIOR);
            
            aI.reshape(4 * v2d * n_colloc * neI); // A prolonged to interior edges
            
            f->action(a, aI);
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

        face_prol->action(u, uI);
        Flx->action(uI, uI);
        face_prol->t(uI, divF);
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
        /// on element el and collocation point (i, j): \f$A^{d}_{k,\ell}=A(k, \ell, d, i, j, el)\f$`
        /// @param constant_coefficient if the coefficient is constant in the domain or if coefficient varies spatially.
        /// @param quad quadrature rule. If ApproxQuadrature, then @a quad is not referenced.
        AdvectionHomogeneousBC(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        ~AdvectionHomogeneousBC() = default;

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

        face_prol = make_face_prolongator(n_var, mesh, basis, Edge::BOUNDARY);

        dvec aB;
        if (constant_coefficient)
        {
            aB.reshape(2 * v2d);

            for (int i=0; i < 2*v2d; ++i)
                aB[i] = a[i];
        }
        else
        {
            auto f = make_face_prolongator(2*v2d, mesh, basis, Edge::BOUNDARY);
            
            aB.reshape(4 * v2d * n_colloc * neB); // A prolonged to boundary edges
            
            f->action(a, aB);
        }

        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, Edge::BOUNDARY, basis, aB, constant_coefficient, -1.0, -0.5, quad));
        
        uB.reshape(2 * n_var * n_colloc * neB);
    }

    template <bool ApproxQuad>
    void AdvectionHomogeneousBC<ApproxQuad>::action(const double * u, double * divF) const
    {
        face_prol->action(u, uB);
        Flx->action(uB, uB);
        face_prol->t(uB, divF);
    }
} // namespace dg

#endif