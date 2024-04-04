#include "Laplacian.hpp"

namespace dg
{
    template <>
    Laplacian<true>::Laplacian(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : dim{2},
          n_basis{basis->n},
          n_elem{mesh.n_elem()},
          n_quad{0},
          D(n_basis, n_basis),
          Dt(n_basis, n_basis),
          _op(3 * n_basis * n_basis * n_elem)
    {
        lagrange_basis_deriv(D, basis->n, basis->x, basis->n, basis->x);
        for (int i=0; i < n_basis; ++i)
            for (int j = 0; j < n_basis; ++j)
                Dt(i, j) = D(j, i);
        
        const int nc2d = n_basis * n_basis;
        auto op = reshape(_op, 3, nc2d, n_elem);
        
        auto& metrics = mesh.element_metrics(basis);
        auto J = reshape(metrics.jacobians(), 2, 2, nc2d, n_elem);
        auto detJ = reshape(metrics.measures(), nc2d, n_elem);

        auto w = reshape(basis->w, basis->n);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < nc2d; ++i)
            {
                const double W = w(i / n_basis) * w(i % n_basis);
                const double Y_eta = J(1, 1, i, el);
                const double X_eta = J(0, 1, i, el);
                const double Y_xi  = J(1, 0, i, el);
                const double X_xi  = J(0, 0, i, el);

                const double mu = detJ(i, el);

                op(0, i, el) = W * ( Y_eta * Y_eta + X_eta * X_eta) / mu;
                op(1, i, el) = W * (-Y_xi  * Y_eta - X_xi  * X_eta) / mu;
                op(2, i, el) = W * ( Y_xi  * Y_xi  + X_xi  * X_xi ) / mu;
            }
        }
    }

    template <>
    void Laplacian<true>::action(int n_var, const double * u_, double * Lu_) const
    {
        auto u = reshape(u_, n_var, n_basis, n_basis, n_elem);
        auto Lu = reshape(Lu_, n_var, n_basis, n_basis, n_elem);

        auto op = reshape(_op, 3, n_basis, n_basis, n_elem); // (A, B, C)
        Tensor<4, double> F(2, n_basis, n_var, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            // compute gradient of u
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const double A = op(0, i, j, el); // w * (Yeta^2 + Xeta^2) / J
                    const double B = op(1, i, j, el); // w * (-Yxi*Yeta - Xxi*Xeta) / J
                    const double C = op(2, i, j, el); // w * (Yxi^2 + Xxi^2) / J

                    for (int d = 0; d < n_var; ++d)
                    {
                        // compute covariant derivative
                        double Dx = 0.0, Dy = 0.0;
                        for (int k = 0; k < n_basis; ++k)
                        {
                            Dx += Dt(k, i) * u(d, k, j, el);
                            Dy += Dt(k, j) * u(d, i, k, el);
                        }

                        // contravariant flux of gradient
                        F(0, i, d, j) = A * Dx + B * Dy;
                        F(1, j, d, i) = B * Dx + C * Dy;
                    }
                }
            }

            // inner product with gradient of basis
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Au = 0.0;
                        for (int l = 0; l < n_basis; ++l)
                        {
                            Au += D(l, i) * F(0, l, d, j) + D(l, j) * F(1, l, d, i);
                        }
                        Lu(d, i, j, el) += Au;
                    }
                }
            }
        }
    }

    template <>
    Laplacian<false>::Laplacian(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : dim{2},
          n_basis{basis->n},
          n_elem{mesh.n_elem()},
          n_quad{quad ? quad->n : (basis->n + mesh.max_element_order())},
          D(n_quad, n_basis),
          Dt(n_basis, n_quad),
          P(n_quad, n_basis),
          Pt(n_basis, n_quad),
          _op(3 * n_quad * n_quad * n_elem)
    {
        if (quad == nullptr)
            quad = QuadratureRule::quadrature_rule(n_quad);
        
        const int nq2d = n_quad * n_quad;
        auto op = reshape(_op, 3, nq2d, n_elem);

        lagrange_basis_deriv(D, basis->n, basis->x, quad->n, quad->x);
        lagrange_basis(P, basis->n, basis->x, quad->n, quad->x);
        for (int i=0; i < n_quad; ++i)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                Dt(j, i) = D(i, j);
                Pt(j, i) = P(i, j);
            }
        }

        auto& metrics = mesh.element_metrics(quad);
        auto J = reshape(metrics.jacobians(), 2, 2, nq2d, n_elem);
        auto detJ = reshape(metrics.measures(), nq2d, n_elem);

        auto w = reshape(quad->w, quad->n);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < nq2d; ++i)
            {
                const double W = w(i / n_quad) * w(i % n_quad);
                const double Y_eta = J(1, 1, i, el);
                const double X_eta = J(0, 1, i, el);
                const double Y_xi  = J(1, 0, i, el);
                const double X_xi  = J(0, 0, i, el);

                const double mu = detJ(i, el);

                op(0, i, el) = W * ( Y_eta * Y_eta + X_eta * X_eta) / mu;
                op(1, i, el) = W * (-Y_xi  * Y_eta - X_xi  * X_eta) / mu;
                op(2, i, el) = W * ( Y_xi  * Y_xi  + X_xi  * X_xi ) / mu;
            }
        }
    }

    template <>
    void Laplacian<false>::action(int n_var, const double * u_, double * Lu_) const
    {
        auto u = reshape(u_, n_var, n_basis, n_basis, n_elem);
        auto Lu = reshape(Lu_, n_var, n_basis, n_basis, n_elem);

        auto op = reshape(_op, 3, n_quad, n_quad, n_elem); // (A, B, C)
        Tensor<4, double> F(2, n_quad, n_var, n_quad);

        dcube PxU(n_basis, n_var, n_quad);
        dcube DxU(n_basis, n_var, n_quad);
        auto DxF = reshape(DxU, n_quad, n_var, n_basis);
        auto PxG = reshape(PxU, n_quad, n_var, n_basis);

        for (int el = 0; el < n_elem; ++el)
        {
            // compute covariant derivatives
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_basis; ++l)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double pu = 0.0, du = 0.0;
                        for (int k = 0; k < n_basis; ++k)
                        {
                            const double uk = u(d, k, l, el);
                            pu += Pt(k, i) * uk;
                            du += Dt(k, i) * uk;
                        }
                        PxU(l, d, i) = pu;
                        DxU(l, d, i) = du;
                    }
                }
            }

            for (int j = 0; j < n_quad; ++j)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    const double A = op(0, i, j, el); // w * (Yeta^2 + Xeta^2) / J
                    const double B = op(1, i, j, el); // w * (-Yxi*Yeta - Xxi*Xeta) / J
                    const double C = op(2, i, j, el); // w * (Yxi^2 + Xxi^2) / J

                    for (int d = 0; d < n_var; ++d)
                    {
                        // covariant derivatives
                        double Dx = 0.0, Dy = 0.0;
                        for (int l = 0; l < n_basis; ++l)
                        {
                            Dx += Pt(l, j) * DxU(l, d, i);
                            Dy += Dt(l, j) * PxU(l, d, i);
                        }

                        // contravariant flux of gradient
                        F(0, i, d, j) = A * Dx + B * Dy;
                        F(1, i, d, j) = B * Dx + C * Dy;
                    }
                }
            }

            // inner product with gradient of basis
            for (int j = 0; j < n_quad; ++j)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double df = 0.0, pg = 0.0;
                        for (int i = 0; i < n_quad; ++i)
                        {
                            df += D(i, k) * F(0, i, d, j);
                            pg += P(i, k) * F(1, i, d, j);
                        }
                        DxF(j, d, k) = df;
                        PxG(j, d, k) = pg;
                    }
                }
            }

            for (int l = 0; l < n_basis; ++l)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Au = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            Au += P(j, l) * DxF(j, d, k) + D(j, l) * PxG(j, d, k);
                        }
                        Lu(d, k, l, el) += Au;
                    }
                }
            }
        }
    }
} // namespace dg
