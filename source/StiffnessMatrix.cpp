#include "StiffnessMatrix.hpp"

static void setup_geometric_factors(int n_elem, const dg::QuadratureRule * quad, dg::TensorWrapper<5, const double> J, dg::Tensor<4,double>& G)
{
    int n_quad = quad->n;

    for (int el = 0; el < n_elem; ++el)
    {
        for (int j = 0; j < n_quad; ++j)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                const double W = quad->w[i] * quad->w[j];
                const double Y_eta = J(1, 1, i, j, el);
                const double X_eta = J(0, 1, i, j, el);
                const double Y_xi  = J(1, 0, i, j, el);
                const double X_xi  = J(0, 0, i, j, el);

                const double detJ = X_xi * Y_eta - X_eta * Y_xi;
                G(0, i, j, el) =  W * (Y_eta * Y_eta + X_eta * X_eta) / detJ;
                G(1, i, j, el) = -W * (Y_xi  * Y_eta + X_xi  * X_eta) / detJ;
                G(2, i, j, el) =  W * (Y_xi  * Y_xi  + X_xi  * X_xi)  / detJ;
            }
        }
    }
}

namespace dg
{
    template <>
    StiffnessMatrix<true>::StiffnessMatrix(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : n_elem{mesh.n_elem()},
          n_basis{basis->n},
          n_quad{0},
          D(n_basis, n_basis),
          G(3, n_basis, n_basis, n_elem)
    {
        lagrange_basis_deriv(D, n_basis, basis->x, n_basis, basis->x);

        auto& metrics = mesh.element_metrics(basis);
        auto J = reshape(metrics.jacobians(), 2, 2, n_basis, n_basis, n_elem);

        setup_geometric_factors(n_elem, basis, J, G);
    }

    template <>
    StiffnessMatrix<false>::StiffnessMatrix(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : n_elem{mesh.n_elem()},
          n_basis{basis->n},
          n_quad{(quad) ? (quad->n) : (n_basis + mesh.max_element_order())},
          P(n_quad, n_basis),
          D(n_quad, n_basis),
          G(3, n_quad, n_quad, n_elem)
    {
        if (not quad)
            quad = QuadratureRule::quadrature_rule(n_quad);

        lagrange_basis(P, n_basis, basis->x, n_quad, quad->x);
        lagrange_basis_deriv(D, n_basis, basis->x, n_quad, quad->x);

        auto& metrics = mesh.element_metrics(quad);
        auto J = reshape(metrics.jacobians(), 2, 2, n_quad, n_quad, n_elem);

        setup_geometric_factors(n_elem, quad, J, G);
    }

    template <>
    void StiffnessMatrix<true>::action(int n_var, const double * u_, double * Au_) const
    {
        auto u = reshape(u_, n_var, n_basis, n_basis, n_elem);
        auto Au = reshape(Au_, n_var, n_basis, n_basis, n_elem);

        dcube Dx(n_basis, n_var, n_basis);
        dcube Dy(n_basis, n_var, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double dx = 0.0, dy = 0.0;
                        for (int k = 0; k < n_basis; ++k)
                        {
                            dx += D(i, k) * u(d, k, j, el);
                            dy += D(j, k) * u(d, i, k, el);
                        }

                        const double A = G(0, i, j, el);
                        const double B = G(1, i, j, el);
                        const double C = G(2, i, j, el);

                        Dx(i, d, j) = A * dx + B * dy;
                        Dy(j, d, i) = B * dx + C * dy;
                    }
                }
            }

            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double s = 0.0;
                        for (int i = 0; i < n_basis; ++i)
                            s += D(i, k) * Dx(i, d, l) + D(i, l) * Dy(i, d, k);
                        Au(d, k, l, el) += s;
                    }
                }
            }
        }
    }

    template <>
    void StiffnessMatrix<true>::action(const double * u, double * Au) const
    {
        action(1, u, Au);
    }

    template <>
    void StiffnessMatrix<false>::action(int n_var, const double * u_, double * Au_) const
    {
        auto u = reshape(u_, n_var, n_basis, n_basis, n_elem);
        auto Au = reshape(Au_, n_var, n_basis, n_basis, n_elem);

        dcube Pu(n_quad, n_var, n_quad);
        dcube Du(n_quad, n_var, n_quad);
        Tensor<4, double> F(2, n_quad, n_var, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double DxU=0.0, PxU=0.0;
                        for (int k = 0; k < n_basis; ++k)
                        {
                            const double ukl = u(d, k, l, el);
                            DxU += D(i, k) * ukl;
                            PxU += P(i, k) * ukl;
                        }
                        Du(l, d, i) = DxU;
                        Pu(l, d, i) = PxU;
                    }
                }
            }

            for (int i = 0; i < n_quad; ++i)
            {
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Dx = 0.0, Dy = 0.0;
                        for (int l = 0; l < n_basis; ++l)
                        {
                            Dx += P(j, l) * Du(l, d, i);
                            Dy += D(j, l) * Pu(l, d, i);
                        }

                        const double A = G(0, i, j, el);
                        const double B = G(1, i, j, el);
                        const double C = G(2, i, j, el);

                        F(0, j, d, i) = A * Dx + B * Dy;
                        F(1, j, d, i) = B * Dx + C * Dy;
                    }
                }
            }

            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double PyF = 0.0, DyF = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            DyF += D(j, l) * F(0, j, d, i);
                            PyF += P(j, l) * F(1, j, d, i);
                        }
                        Du(i, d, l) = DyF;
                        Pu(i, d, l) = PyF;
                    }
                }
            }

            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double s = 0.0;
                        for (int i = 0; i < n_quad; ++i)
                            s += D(i, k) * Pu(i, d, l) + P(i, k) * Du(i, d, l);
                        Au(d, k, l, el) += s;
                    }
                }
            }
        }
    }

    template <>
    void StiffnessMatrix<false>::action(const double * u, double * Au) const
    {
        action(1, u, Au);
    }
} // namespace dg
