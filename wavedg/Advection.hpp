#ifndef WDG_ADVECTION_HPP
#define WDG_ADVECTION_HPP

#include "wdg_config.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
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

        Advection(int n_var, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

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
        const int neI = mesh.n_edges(FaceType::INTERIOR);
        const int n_colloc = basis->n;

        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(n_var, mesh, basis, FaceType::INTERIOR);

        dvec aI;
        if (constant_coefficient)
        {
            aI.reshape(2 * v2d);

            for (int i=0; i < 2*v2d; ++i)
                aI[i] = a[i];
        }
        else
        {
            auto f = make_face_prolongator(2*v2d, mesh, basis, FaceType::INTERIOR);
            
            aI.reshape(4 * v2d * n_colloc * neI); // A prolonged to interior edges
            
            f->action(a, aI);
        }
        
        div.reset(new Div<ApproxQuad>(n_var, mesh, basis, a, constant_coefficient, quad));

        // take negative a,b since we are computing -div rather than div.
        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, FaceType::INTERIOR, basis, aI, constant_coefficient, -1.0, -0.5, quad));
        
        uI.reshape(2 * n_var * n_colloc * neI);
    }

    template <bool ApproxQuad>
    Advection<ApproxQuad>::Advection(int n_var_, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad)
        : n_var(n_var_)
    {
        const int neI = mesh.n_faces(FaceType::INTERIOR);
        const int n_colloc = basis->n;

        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(n_var, mesh, basis, FaceType::INTERIOR);

        dvec aI;
        if (constant_coefficient)
        {
            aI.reshape(v2d);

            for (int i=0; i < v2d; ++i)
                aI(i) = a[i];
        }
        else
        {
            auto f = make_face_prolongator(v2d, mesh, basis, FaceType::INTERIOR);

            aI.reshape(2 * v2d * neI);

            f->action(a, aI);
        }

        div.reset(new Div<ApproxQuad>(n_var, mesh, basis, a, constant_coefficient, quad));

        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, FaceType::INTERIOR, basis, aI, constant_coefficient, -1.0, -0.5, quad));

        uI.reshape(2 * n_var * neI);
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
        const int dim;
        const int n_var;

        std::unique_ptr<FaceProlongator> face_prol;
        std::unique_ptr<Operator> Flx;

        ivec b;
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

        AdvectionHomogeneousBC(int n_var, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad=nullptr);

        ~AdvectionHomogeneousBC() = default;

        /// @brief adds boundary conditions
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem)
        /// @param divF shape (n_var, n_colloc, n_colloc, n_elem). On exit, divF <- divF + BC
        void action(const double * u, double * divF) const override;
    };

    template <bool ApproxQuad>
    AdvectionHomogeneousBC<ApproxQuad>::AdvectionHomogeneousBC(int nv, const Mesh2D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad)
        : dim(2), n_var(nv)
    {
        const int neB = mesh.n_edges(FaceType::BOUNDARY);
        const int n_colloc = basis->n;
        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(n_var, mesh, basis, FaceType::BOUNDARY);

        dvec aB;
        if (constant_coefficient)
        {
            aB.reshape(2 * v2d);

            for (int i=0; i < 2*v2d; ++i)
                aB[i] = a[i];
        }
        else
        {
            auto f = make_face_prolongator(2*v2d, mesh, basis, FaceType::BOUNDARY);
            
            aB.reshape(4 * v2d * n_colloc * neB); // A prolonged to boundary edges
            
            f->action(a, aB);
        }

        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, FaceType::BOUNDARY, basis, aB, constant_coefficient, -1.0, -0.5, quad));
        
        uB.reshape(2 * n_var * n_colloc * neB);
    }

    template <bool ApproxQuad>
    AdvectionHomogeneousBC<ApproxQuad>::AdvectionHomogeneousBC(int nv, const Mesh1D& mesh, const QuadratureRule * basis, const double * a, bool constant_coefficient, const QuadratureRule * quad)
        : dim(1), n_var(nv)
    {
        const int neB = mesh.n_faces(FaceType::BOUNDARY);
        const int v2d = n_var * n_var;

        face_prol = make_face_prolongator(n_var, mesh, basis, FaceType::BOUNDARY);

        dvec aB;
        if (constant_coefficient)
        {
            aB.reshape(v2d);

            for (int i = 0; i < v2d; ++i)
                aB(i) = a[i];
        }
        else
        {
            auto f = make_face_prolongator(v2d, mesh, basis, FaceType::BOUNDARY);

            aB.reshape(2 * v2d * neB);

            f->action(a, aB);
        }

        Flx.reset(new EdgeFlux<ApproxQuad>(n_var, mesh, FaceType::BOUNDARY, basis, aB, constant_coefficient, -1.0, -0.5, quad));

        uB.reshape(2 * n_var * neB);

        b.reshape(neB);
        for (int e = 0; e < neB; ++e)
        {
            auto& face = mesh.face(e, FaceType::BOUNDARY);
            if (face.elements[1] >= 0)
                b(e) = 0; // left boundary
            else
                b(e) = 1; // right boundary
        }
    }

    template <bool ApproxQuad>
    void AdvectionHomogeneousBC<ApproxQuad>::action(const double * u, double * divF) const
    {
        face_prol->action(u, uB);
        
        if (dim == 1)
        {
            const int neB = b.size();
            auto ub = reshape(uB, n_var, 2, neB);
            for (int e = 0; e < neB; ++e)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    ub(d, b(e), e) = 0.0;
                }
            }
        }
        
        Flx->action(uB, uB);
        face_prol->t(uB, divF);
    }
} // namespace dg

#endif