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
} // namespace dg
