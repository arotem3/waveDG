#include "DivF.hpp"

namespace dg
{
    template <>
    DivF1D<true>::DivF1D(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem()),
          quad(basis),
          D(n_colloc, n_colloc),
          Uq(n_var),
          Fq(n_var),
          F(n_colloc, n_var)
    {
        auto x_ = mesh.element_metrics(basis).physical_coordinates();
        x = reshape(x_, n_colloc, n_elem);

        lagrange_basis_deriv(D, n_colloc, basis->x, n_colloc, basis->x);
    }

    template <>
    DivF1D<false>::DivF1D(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const QuadratureRule * quad_)
        : n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem()),
          quad{quad_ ? quad_ : QuadratureRule::quadrature_rule(2*n_colloc)},
          D(quad->n, n_colloc),
          Pt(n_colloc, quad->n),
          Uq(n_var),
          Fq(n_var),
          F(quad->n, n_var)
    {
        const int n_quad = quad->n;
        dmat P(n_quad, n_colloc);

        lagrange_basis(P, n_colloc, basis->x, n_quad, quad->x);
        for (int j = 0; j < n_colloc; ++j)
            for (int i = 0; i < n_quad; ++i)
                Pt(j, i) = P(i, j);
        
        lagrange_basis_deriv(D, n_colloc, basis->x, n_quad, quad->x);
    }

    template <>
    DivF2D<true>::DivF2D(int nvar, const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule *quad_)
        : n_var{nvar},
          n_colloc{basis->n},
          n_elem{mesh.n_elem()},
          quad{basis},
          n_quad{basis->n},
          D(n_colloc, n_colloc),
          Uq(n_var),
          Fq(n_var, 2),
          F(2, n_colloc, n_var, n_colloc)
    {
        lagrange_basis_deriv(D, n_colloc, basis->x, n_colloc, basis->x);

        auto& metrics = mesh.element_metrics(basis);
        X = reshape(metrics.physical_coordinates(), 2, n_colloc, n_colloc, n_elem);
        J = reshape(metrics.jacobians(), 2, 2, n_colloc, n_colloc, n_elem);
    }

    template <>
    DivF2D<false>::DivF2D(int nvar, const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule *quad_)
        : n_var{nvar},
          n_colloc{basis->n},
          n_elem{mesh.n_elem()},
          quad{quad_ ? quad_ : QuadratureRule::quadrature_rule(2*n_colloc)},
          n_quad{quad->n},
          D(n_quad, n_colloc),
          Dt(n_colloc, n_quad),
          P(n_quad, n_colloc),
          Pt(n_colloc, n_quad),
          Uq(n_var),
          Fq(n_var, 2),
          F(2, n_quad, n_var, n_quad),
          work1(n_var * n_quad * n_colloc),
          work2(n_var * n_quad * n_colloc)
    {
        lagrange_basis(P, n_colloc, basis->x, n_quad, quad->x);
        lagrange_basis_deriv(D, n_colloc, basis->x, n_quad, quad->x);

        for (int i = 0; i < n_quad; ++i)
        {
            for (int j = 0; j < n_colloc; ++j)
            {
                Pt(j, i) = P(i, j);
                Dt(j, i) = D(i, j);
            }
        }

        auto& metrics = mesh.element_metrics(quad);
        X = reshape(metrics.physical_coordinates(), 2, n_quad, n_quad, n_elem);
        J = reshape(metrics.jacobians(), 2, 2, n_quad, n_quad, n_elem);
    }
} // namespace dg
