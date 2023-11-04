#include "Div.hpp"

namespace dg
{
    template <>
    Div<true>::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * _A, bool constant_coefficient, const QuadratureRule * quad)
        : n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem()),
          D(n_colloc, n_colloc),
          _op(2 * n_var * n_var * n_colloc * n_colloc * n_elem),
          Fq(2, n_colloc, n_var, n_colloc)
    {
        lagrange_basis_deriv(D.data(), n_colloc, basis->x, n_colloc, basis->x);

        const int c2d = n_colloc * n_colloc;

        const double * _J = mesh.element_jacobians(basis);
        auto J = reshape(_J, 2, 2, c2d, n_elem);
        auto w = reshape(basis->w, basis->n);

        auto op = reshape(_op.data(), 2, n_var, n_var, c2d, n_elem);
        auto a = reshape(_A, n_var, n_var, 2, c2d, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            const int _el = constant_coefficient ? 0 : el; // index for a

            for (int i = 0; i < c2d; ++i)
            {
                const int _i = constant_coefficient ? 0 : i; // index for a

                const double W = w[i/n_colloc] * w[i%n_colloc];
                const double Y_eta = J(1, 1, i, el) * W;
                const double X_eta = J(0, 1, i, el) * W;
                const double Y_xi  = J(1, 0, i, el) * W;
                const double X_xi  = J(0, 0, i, el) * W;

                for (int d = 0; d < n_var; ++d)
                {
                    for (int c = 0; c < n_var; ++c)
                    {
                        const double a0 = a(c, d, 0, _i, _el);
                        const double a1 = a(c, d, 1, _i, _el);

                        op(0, d, c, i, el) =  Y_eta * a0 - X_eta * a1;
                        op(1, d, c, i, el) = -Y_xi  * a0 + X_xi  * a1;
                    }
                }
            }
        }
    }

    template <>
    void Div<true>::action(const double * u_, double * du_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_colloc, n_elem);

        auto op = reshape(_op.data(), 2, n_var, n_var, n_colloc, n_colloc, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            // compute contravariant flux
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Fd = 0.0;
                        double Gd = 0.0;
                        for (int c = 0; c < n_var; ++c)
                        {
                            const double uij = u(c, i, j, el);
                            Fd += op(0, c, d, i, j, el) * uij;
                            Gd += op(1, c, d, i, j, el) * uij;
                        }
                        Fq(0, i, d, j) = Fd;
                        Fq(1, j, d, i) = Gd;
                    }
                }
            }

            // integral (F, grad v)
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double divF = 0.0;
                        for (int l = 0; l < n_colloc; ++l)
                        {
                            divF += D(l, i) * Fq(0, l, d, j) + D(l, j) * Fq(1, l, d, i);
                        }
                        du(d, i, j, el) += divF;
                    }
                }
            }
        }
    }

    template <>
    Div<false>::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad)
        : n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem())
    {
        if (quad == nullptr)
        {
            int p = (n_colloc - 2) + (n_colloc - 1) + mesh.max_element_order();
            if (not constant_coefficient)
                p += n_colloc - 1;
            p = 1 + p/2;

            quad = QuadratureRule::quadrature_rule(p);
        }
        n_quad = quad->n;

        D.reshape(n_quad, n_colloc);
        Dt.reshape(n_colloc, n_quad);

        P.reshape(n_quad, n_colloc);
        Pt.reshape(n_colloc, n_quad);
        
        _op.reshape(2 * n_var * n_var * n_quad * n_quad * n_elem);
        
        Uq.reshape(n_var, n_quad, n_quad);
        Fq.reshape(2, n_var, n_quad, n_quad);
        
        Df.reshape(n_quad, n_var, n_colloc);
        Pg.reshape(n_quad, n_var, n_colloc);
        Pu = reshape(Df.data(), n_colloc, n_var, n_quad);

        lagrange_basis(P, n_colloc, basis->x, n_quad, quad->x);
        lagrange_basis_deriv(D, n_colloc, basis->x, n_quad, quad->x);

        for (int j = 0; j < n_colloc; ++j)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                Dt(j, i) = D(i, j);
                Pt(j, i) = P(i, j);
            }
        }

        const double * _J = mesh.element_jacobians(quad);
        auto J = reshape(_J, 2, 2, n_quad, n_quad, n_elem);
        auto op = reshape(_op.data(), 2, n_var, n_var, n_quad, n_quad, n_elem);
        auto w = reshape(quad->w, n_quad);

        if (constant_coefficient)
        {
            auto a = reshape(A, n_var, n_var, 2);
            for (int el = 0; el < n_elem; ++el)
            {
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int i = 0; i < n_quad; ++i)    
                    {
                        const double W = w(i) * w(j);
                        const double X_xi  = J(0, 0, i, j, el) * W;
                        const double Y_xi  = J(1, 0, i, j, el) * W;
                        const double X_eta = J(0, 1, i, j, el) * W;
                        const double Y_eta = J(1, 1, i, j, el) * W;

                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                double aq0 = a(d, c, 0);
                                double aq1 = a(d, c, 1);

                                op(0, c, d, i, j, el) =  Y_eta * aq0 - X_eta * aq1;
                                op(1, c, d, i, j, el) = -Y_xi  * aq0 + X_xi  * aq1;
                            }
                        }
                    }
                }
            }
        }
        else
        {
            auto a = reshape(A, n_var, n_var, 2, n_colloc, n_colloc, n_elem);
            Tensor<5, double> PA(2, n_var, n_var, n_colloc, n_quad);
            for (int el = 0; el < n_elem; ++el)
            {
                // a(i, k) <- P(i, l) * a(l, k)
                for (int k = 0; k < n_colloc; ++k)
                {
                    for (int i = 0; i < n_quad; ++i)
                    {
                        for (int c = 0; c < n_var; ++c)
                        {
                            for (int d = 0; d < n_var; ++d)
                            {
                                double aik0 = 0.0;
                                double aik1 = 0.0;
                                for (int l = 0; l < n_colloc; ++l)
                                {
                                    aik0 += Pt(l, i) * a(d, c, 0, l, k, el);
                                    aik1 += Pt(l, i) * a(d, c, 1, l, k, el);
                                }
                                PA(0, c, d, k, i) = aik0;
                                PA(1, c, d, k, i) = aik1;
                            }
                        }
                    }
                }

                // aq(i, j) <- P(j, k) * a(i, k)
                for (int j = 0; j < n_quad; ++j)
                {
                    for (int i = 0; i < n_quad; ++i)    
                    {
                        const double W = w(i) * w(j);
                        const double X_xi  = J(0, 0, i, j, el) * W;
                        const double Y_xi  = J(1, 0, i, j, el) * W;
                        const double X_eta = J(0, 1, i, j, el) * W;
                        const double Y_eta = J(1, 1, i, j, el) * W;

                        for (int d = 0; d < n_var; ++d)
                        {
                            for (int c = 0; c < n_var; ++c)
                            {
                                double aq0 = 0.0;
                                double aq1 = 0.0;
                                for (int k = 0; k < n_colloc; ++k)
                                {
                                    aq0 += Pt(k, j) * PA(0, c, d, k, i);
                                    aq1 += Pt(k, j) * PA(1, c, d, k, i);
                                }

                                op(0, c, d, i, j, el) =  Y_eta * aq0 - X_eta * aq1;
                                op(1, c, d, i, j, el) = -Y_xi  * aq0 + X_xi  * aq1;
                            }
                        }
                    }
                }
            }
        }
    }

    template <>
    void Div<false>::action(const double * u_, double * du_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_colloc, n_elem);

        auto op = reshape(_op.data(), 2, n_var, n_var, n_quad, n_quad, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate u on quadrature points
            // Uq(i, j) = P(i, k) * P(j, l) * u(k, l)
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_colloc; ++l)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double pu = 0.0;
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            pu += Pt(k, i) * u(d, k, l, el);
                        }
                        Pu(l, d, i) = pu;
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
                        for (int l = 0; l < n_colloc; ++l)
                        {
                            uq += Pt(l, j) * Pu(l, d, i);
                        }
                        Uq(d, i, j) = uq;
                    }
                }
            }

            // compute contravariant flux
            for (int j = 0; j < n_quad; ++j)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Fd = 0.0;
                        double Gd = 0.0;
                        for (int c = 0; c < n_var; ++c)
                        {
                            Fd += op(0, c, d, i, j, el) * Uq(c, i, j);
                            Gd += op(1, c, d, i, j, el) * Uq(c, i, j);
                        }
                        Fq(0, d, i, j) = Fd;
                        Fq(1, d, i, j) = Gd;
                    }
                }
            }

            // integral (F, grad v)
            // (F, grad v)(k, l) = F(i, j)*D(i, k)*P(j, l) + G(i, j)*P(i, k)*D(j, l)
            for (int j = 0; j < n_quad; ++j)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double df = 0.0;
                        double pg = 0.0;
                        for (int i = 0; i < n_quad; ++i)
                        {
                            df += D(i, k) * Fq(0, d, i, j);
                            pg += P(i, k) * Fq(1, d, i, j);
                        }
                        Df(j, d, k) = df;
                        Pg(j, d, k) = pg;
                    }
                }
            }


            for (int l = 0; l < n_colloc; ++l)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double divF = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            divF += P(j, l) * Df(j, d, k) + D(j, l) * Pg(j, d, k);
                        }
                        du(d, k, l, el) += divF;
                    }
                }
            }
        }
    }

} // namespace dg
