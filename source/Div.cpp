#include "Div.hpp"

namespace dg
{
    Div::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * _A, bool constant_coefficient)
        : n_var(nvar),
          n_colloc(basis->n),
          n_quad(basis->n),
          n_elem(mesh.n_elem()),
          approx_quadrature(true),
          quad(basis),
          D(n_quad, n_colloc),
          P(),
          _op(2 * n_var * n_var * n_quad * n_quad * n_elem),
          Fq(n_var, n_quad, n_quad, 2),
          PF(n_var, n_quad, n_quad)
    {
        lagrange_basis_deriv(D.data(), n_colloc, basis->x, n_quad, basis->x);

        const int nc2d = n_colloc * n_colloc;
        const int nv2d = n_var * n_var;

        const double * _J = mesh.element_jacobians(basis);
        auto J = reshape(_J, 2, 2, nc2d, n_elem);

        auto op = reshape(_op.data(), nv2d, 2, nc2d, n_elem);

        auto a = reshape(_A, nv2d, 2, nc2d, n_elem);

        const double * w = quad->w;

        for (int el = 0; el < n_elem; ++el)
        {
            const int _el = constant_coefficient ? 0 : el; // index for a

            for (int i = 0; i < nc2d; ++i)
            {
                const int _i = constant_coefficient ? 0 : i; // index for a

                const double W = w[i/n_colloc] * w[i%n_colloc];

                for (int d = 0; d < nv2d; ++d)
                {
                    const double a0 = a(d, 0, _i, _el);
                    const double a1 = a(d, 1, _i, _el);

                    op(d, 0, i, el) = ( J(1,1, i, el) * a0 - J(0,1, i, el) * a1) * W;
                    op(d, 1, i, el) = (-J(1,0, i, el) * a0 + J(0,0, i, el) * a1) * W;
                }
            }
        }
    }

    Div::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad_, const double * A, bool constant_coefficient)
        : n_var(nvar),
          n_colloc(basis->n),
          n_quad(quad_->n),
          n_elem(mesh.n_elem()),
          approx_quadrature(false),
          quad(quad_),
          D(n_quad, n_colloc),
          P(n_quad, n_colloc),
          _op(2 * n_var * n_var * n_quad * n_quad * n_elem),
          Uq(n_var, n_quad, n_quad),
          Fq(n_var, n_quad, n_quad, 2),
          PF(n_var, n_colloc, n_quad)
    {
        lagrange_basis(P.data(), n_colloc, basis->x, n_quad, quad->x);
        lagrange_basis_deriv(D.data(), n_colloc, basis->x, n_quad, quad->x);

        const int nc2d = n_colloc * n_colloc;
        const int nq2d = n_quad * n_quad;
        const int nv2d = n_var * n_var;

        const double * _J = mesh.element_jacobians(quad);
        auto J = reshape(_J, 2, 2, nq2d, n_elem);
        auto op = reshape(_op.data(), nv2d, 2, nq2d, n_elem);
        auto a = reshape(A, nv2d, 2, nc2d, n_elem);
        auto w = reshape(quad->w, n_quad);

        dmat Aq(nv2d, 2);

        for (int el = 0; el < n_elem; ++el)
        {
            const int _el = constant_coefficient ? 0 : el; // index for a

            for (int i = 0; i < nq2d; ++i)
            {
                // evaluate a on quadrature point
                for (int s = 0; s < 2; ++s)
                {
                    for (int d = 0; d < nv2d; ++d)
                    {
                        double aq = 0.0;
                        for (int j = 0; j < nc2d; ++j)
                        {
                            const int _j = constant_coefficient ? 0 : j;
                            const double b = P(i%n_quad, j%n_colloc) * P(i/n_quad, j/n_colloc);
                            aq += a(d, s, _j, _el) * b;
                        }
                        Aq(d, s) = aq;
                    }
                }

                const double W = w(i/n_quad) * w(i%n_quad);

                for (int d = 0; d < nv2d; ++d)
                {
                    const double a0 = Aq(d, 0);
                    const double a1 = Aq(d, 1);

                    op(d, 0, i, el) = ( J(1,1, i, el) * a0 - J(0,1, i, el) * a1) * W;
                    op(d, 1, i, el) = (-J(1,0, i, el) * a0 + J(0,0, i, el) * a1) * W;
                }
            }
        }
    }

    void Div::operator()(const double * u_, double * du_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_colloc, n_elem);

        auto op = reshape(_op.data(), n_var, n_var, 2, n_quad, n_quad, n_elem);

        if (approx_quadrature)
        {
            for (int el = 0; el < n_elem; ++el)
            {
                // compute contravariant flux
                for (int s = 0; s < 2; ++s)
                {
                    for (int j = 0; j < n_colloc; ++j)
                    {
                        for (int i = 0; i < n_colloc; ++i)
                        {
                            for (int d = 0; d < n_var; ++d)
                            {
                                double Fd = 0.0;
                                for (int k = 0; k < n_var; ++k)
                                {
                                    Fd += op(d, k, s, i, j, el) * u(k, i, j, el);
                                }
                                Fq(d, i, j, s) = Fd;
                            }
                        }
                    }
                }

                // integral -(F, grad v)
                for (int j = 0; j < n_colloc; ++j)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double divF = 0.0;
                            for (int l = 0; l < n_colloc; ++l)
                            {
                                divF -= D(l, i) * Fq(d, l, j, 0) + D(l, j) * Fq(d, i, l, 1);
                            }
                            du(d, i, j, el) = divF;
                        }
                    }
                }
            }
        }
        else
        {
            auto& Pu = PF;
            auto& divF = Uq; // use same memory
            
            for (int el = 0; el < n_elem; ++el)
            {
            // evaluate u on quadrature points
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double pu = 0.0;
                            for (int l = 0; l < n_colloc; ++l)
                            {
                                pu += P(j, l) * u(d, k, l, el);
                            }
                            Pu(d, k, j) = pu;
                        }
                    }
                }
                
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int i = 0; i < n_quad; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double uq = 0.0;
                            for (int k = 0; k < n_colloc; ++k)
                            {
                                uq += P(i, k) * Pu(d, k, j);
                            }
                            Uq(d, i, j) = uq;
                        }
                    }
                }

            // compute contravariant flux
                for (int s = 0; s < 2; ++s)
                {
                    for (int j = 0; j < n_quad; ++j)
                    {
                        for (int i = 0; i < n_quad; ++i)
                        {
                            for (int d = 0; d < n_var; ++d)
                            {
                                double Fd = 0.0;
                                for (int k = 0; k < n_var; ++k)
                                {
                                    Fd += op(d, k, s, i, j, el) * Uq(k, i, j);
                                }
                                Fq(d, i, j, s) = Fd;
                            }
                        }
                    }
                }

            // integral -(F, grad v)
            // -(F, grad v)(k, l) = -F(i, j, 0)*D(i, k)*P(j, l) - F(i, j, 1)*P(i, k)*D(j, l)
                // DF0(k, j) = D(i, k) * F(i, j, 0)
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double DF0 = 0.0;
                            for (int i = 0; i < n_quad; ++i)
                            {
                                DF0 += D(i, k) * Fq(d, i, j, 0);
                            }
                            PF(d, k, j) = DF0;
                        }
                    }
                }

                // PDF0(k, l) = P(j, l) * DF(k, j)
                // divF(k, l) = -PDF0(k, l)
                for (int l = 0; l < n_colloc; ++l)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double PDF0 = 0.0;
                            for (int j = 0; j < n_quad; ++j)
                            {
                                PDF0 += PF(d, k, j) * P(j, l);
                            }
                            divF(d, k, l) = -PDF0;
                        }
                    }
                }

                // PF1(k, j) = P(i, k) * F(i, j, 1)
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double PF1 = 0.0;
                            for (int i = 0; i < n_quad; ++i)
                            {
                                PF1 += P(i, k) * Fq(d, i, j, 1);
                            }
                            PF(d, k, j) = PF1;
                        }
                    }
                }

                // DPF1(k, l) = D(j, l) * PF1(k, j)
                // divF(k, l) -= DPF1(k, l)
                for (int l = 0; l < n_colloc; ++l)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double DPF1 = 0.0;
                            for (int j = 0; j < n_quad; ++j)
                            {
                                DPF1 += PF(d, k, j) * D(j, l);
                            }
                            divF(d, k, l) -= DPF1;
                        }
                    }
                }

            // copy divF to du
                for (int l = 0; l < n_colloc; ++l)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            du(d, k, l, el) = divF(d, k, l);
                        }
                    }
                }
            }
        }
    }
} // namespace dg
