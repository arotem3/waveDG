#include "Nabla.hpp"

namespace dg
{
    template <>
    Nabla<true>::Nabla(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
        : n_elem{mesh.n_elem()},
          n_basis{basis->n},
          D(n_basis, n_basis)
    {
        lagrange_basis_deriv(D, n_basis, basis->x, n_basis, basis->x);

        auto& metrics = mesh.element_metrics(basis);
        J = reshape(metrics.jacobians(), 2, 2, n_basis, n_basis, n_elem);
        w = reshape(basis->w, n_basis);
    }

    template <>
    void Nabla<true>::grad(const double * u_, double * grad_u_) const
    {
        auto u = reshape(u_, n_basis, n_basis, n_elem);
        auto grad_u = reshape(grad_u_, 2, n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    double Dx = 0.0, Dy = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        Dx += D(i, k) * u(k, j, el);
                        Dy += D(j, k) * u(i, k, el);
                    }

                    const double W = w[i] * w[j];
                    const double Y_eta = W * J(1, 1, i, j, el);
                    const double X_eta = W * J(0, 1, i, j, el);
                    const double Y_xi  = W * J(1, 0, i, j, el);
                    const double X_xi  = W * J(0, 0, i, j, el);

                    grad_u(0, i, j, el) =  Y_eta * Dx - Y_xi * Dy;
                    grad_u(1, i, j, el) = -X_eta * Dx + X_xi * Dy;
                }
            }
        }
    }

    template <>
    void Nabla<true>::div(const double * u_, double * div_u_) const
    {
        auto u = reshape(u_, 2, n_basis, n_basis, n_elem);
        auto div_u = reshape(div_u_, n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    const double W = w[i] * w[j];
                    const double Y_eta = W * J(1, 1, i, j, el);
                    const double X_eta = W * J(0, 1, i, j, el);
                    const double Y_xi  = W * J(1, 0, i, j, el);
                    const double X_xi  = W * J(0, 0, i, j, el);

                    double s = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        double f =  Y_eta * u(0, k, j, el) - X_eta * u(1, k, j, el);
                        double g = -Y_xi  * u(0, i, k, el) + X_xi  * u(1, i, k, el);
                        s += D(i, k) * f + D(j, k) * g;
                    }

                    div_u(i, j, el) = s;
                }
            }
        }
    }

    template <>
    void Nabla<true>::xycurl(const double * u_, double * curl_u_) const
    {
        auto u = reshape(u_, 2, n_basis, n_basis, n_elem);
        auto curl_u = reshape(curl_u_, n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    double Ux = 0.0, Uy = 0.0, Vx = 0.0, Vy = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        Ux += D(i, k) * u(0, k, j, el);
                        Uy += D(j, k) * u(0, i, k, el);

                        Vx += D(i, k) * u(1, k, j, el);
                        Vy += D(j, k) * u(1, i, k, el);
                    }

                    const double W = w[i] * w[j];
                    const double Y_eta = W * J(1, 1, i, j, el);
                    const double X_eta = W * J(0, 1, i, j, el);
                    const double Y_xi  = W * J(1, 0, i, j, el);
                    const double X_xi  = W * J(0, 0, i, j, el);

                    const double uy = -X_eta * Ux + X_xi * Uy;
                    const double vx =  Y_eta * Vx - Y_xi * Vy;

                    curl_u(i, j, el) = vx - uy;
                }
            }
        }
    }

    template <>
    void Nabla<true>::zcurl(const double * u_, double * curl_u_) const
    {
        auto u = reshape(u_, n_basis, n_basis, n_elem);
        auto curl_u = reshape(curl_u_, 2, n_basis, n_basis, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_basis; ++j)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    double Dx = 0.0, Dy = 0.0;
                    for (int k = 0; k < n_basis; ++k)
                    {
                        Dx += D(i, k) * u(k, j, el);
                        Dy += D(j, k) * u(i, k, el);
                    }

                    const double W = w[i] * w[j];
                    const double Y_eta = W * J(1, 1, i, j, el);
                    const double X_eta = W * J(0, 1, i, j, el);
                    const double Y_xi  = W * J(1, 0, i, j, el);
                    const double X_xi  = W * J(0, 0, i, j, el);

                    curl_u(0, i, j, el) = -X_eta * Dx + X_xi * Dy;
                    curl_u(1, i, j, el) = -Y_eta * Dx + Y_xi * Dy;
                }
            }
        }
    }
} // namespace dg
