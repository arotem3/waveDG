#include "Div.hpp"

namespace dg
{
    template <>
    Div<true>::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * _A, bool constant_coefficient, const QuadratureRule * quad)
        : dim(2),
          n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem()),
          D(n_colloc, n_colloc),
          _op(2 * n_var * n_var * n_colloc * n_colloc * n_elem),
          Fq(2 * n_colloc * n_var * n_colloc)
    {
        lagrange_basis_deriv(D, n_colloc, basis->x, n_colloc, basis->x);

        const int c2d = n_colloc * n_colloc;

        const double * _J = mesh.element_metrics(basis).jacobians();
        auto J = reshape(_J, 2, 2, c2d, n_elem);
        auto w = reshape(basis->w, basis->n);

        auto op = reshape(_op, 2, n_var, n_var, c2d, n_elem);
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
    Div<true>::Div(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const double * _A, bool constant_coefficient, const QuadratureRule * quad)
        : dim(1),
          n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem()),
          D(n_colloc, n_colloc),
          _op(n_var * n_var * n_colloc * n_elem),
          Fq(n_colloc * n_var)
    {
        lagrange_basis_deriv(D, n_colloc, basis->x, n_colloc, basis->x);

        auto w = reshape(basis->w, n_colloc);

        auto op = reshape(_op, n_var, n_var, n_colloc, n_elem);
        auto a = reshape(_A, n_var, n_var, n_colloc, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            const int _el = constant_coefficient ? 0 : el; // index in a

            for (int i=0; i < n_colloc; ++i)
            {
                const int _i = constant_coefficient ? 0 : i; // index in a

                for (int d = 0; d < n_var; ++d)
                {
                    for (int c = 0; c < n_var; ++c)
                    {
                        op(d, c, i, el) = a(c, d, _i, _el) * w(i);
                    }
                }
            }
        }
    }

    static void div_1d_c(int n_elem, int n_colloc, int n_var, const double * u_, double * du_, const double * _op, double * _F, const dmat& D)
    {
        auto u = reshape(u_, n_var, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_elem);

        auto op = reshape(_op, n_var, n_var, n_colloc, n_elem);
        auto Fq = reshape(_F, n_colloc, n_var);

        for (int el = 0; el < n_elem; ++el)
        {
            // compute flux
            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double Fd = 0.0;
                    for (int c = 0; c < n_var; ++c)
                    {
                        Fd += op(c, d, i, el) * u(c, i, el);
                    }
                    Fq(i, d) = Fd;
                }
            }

            // integral (F, dv/dx)
            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double divF = 0.0;
                    for (int l=0; l < n_colloc; ++l)
                    {
                        divF += D(l, i) * Fq(l, d);
                    }
                    du(d, i, el) += divF;
                }
            }
        }
    }

    static void div_2d_c(int n_elem, int n_colloc, int n_var, const double * u_, double * du_, const double * _op, double * _F, const dmat& D)
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_colloc, n_elem);

        auto op = reshape(_op, 2, n_var, n_var, n_colloc, n_colloc, n_elem);
        auto Fq = reshape(_F, 2, n_colloc, n_var, n_colloc);

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
    void Div<true>::action(const double * u_, double * du_) const
    {
        switch (dim)
        {
        case 1:
            div_1d_c(n_elem, n_colloc, n_var, u_, du_, _op, Fq, D);
            break;
        case 2:
            div_2d_c(n_elem, n_colloc, n_var, u_, du_, _op, Fq, D);
            break;
        default:
            wdg_error("Div<true>::action error: not implemented for specified dimension.");
            break;
        }
    }

    static void op_2d_cc(int n_var, const double * A, const Mesh2D& mesh, const QuadratureRule * quad, double * _op)
    {
        const int n_quad = quad->n;
        const int n_elem = mesh.n_elem();

        auto a = reshape(A, n_var, n_var, 2);
        auto op = reshape(_op, 2, n_var, n_var, n_quad, n_quad, n_elem);

        auto _J = mesh.element_metrics(quad).jacobians();
        auto J = reshape(_J, 2, 2, n_quad, n_quad, n_elem);

        auto w = reshape(quad->w, n_quad);

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
    
    static void op_2d(int n_var, const double * A, const Mesh2D& mesh, const QuadratureRule * quad, double * _op, const dmat& Pt)
    {
        const int n_elem = mesh.n_elem();
        const int n_quad = quad->n;
        const int n_colloc = Pt.shape()[0];

        auto a = reshape(A, n_var, n_var, 2, n_colloc, n_colloc, n_elem);
        auto op = reshape(_op, 2, n_var, n_var, n_quad, n_quad, n_elem);

        auto _J = mesh.element_metrics(quad).jacobians();
        auto J = reshape(_J, 2, 2, n_quad, n_quad, n_elem);

        auto w = reshape(quad->w, n_quad);

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

    template <>
    Div<false>::Div(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad)
        : dim(2),
          n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem())
    {
        if (quad == nullptr)
        {
            // p = (basis degree) + (basis deriv dergree) + (mesh jacobian degree)
            int p = (n_colloc - 2) + (n_colloc - 1) + mesh.max_element_order();
            if (not constant_coefficient)
                p += n_colloc - 1;
            p = 1 + p/2; // quadrature rule is exact for 2*degree-1 polynomials

            quad = QuadratureRule::quadrature_rule(p);
        }
        n_quad = quad->n;

        D.reshape(n_quad, n_colloc);
        Dt.reshape(n_colloc, n_quad);

        P.reshape(n_quad, n_colloc);
        Pt.reshape(n_colloc, n_quad);
        
        _op.reshape(2 * n_var * n_var * n_quad * n_quad * n_elem);
        
        Uq.reshape(n_var * n_quad * n_quad);
        Fq.reshape(2 * n_var * n_quad * n_quad);
        
        Df.reshape(n_quad * n_var * n_colloc);
        Pg.reshape(n_quad * n_var * n_colloc);
        
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

        if (constant_coefficient)
            op_2d_cc(n_var, A, mesh, quad, _op);
        else
            op_2d(n_var, A, mesh, quad, _op, Pt);
    }

    static void op_1d_cc(int n_var, const double * A, const Mesh1D& mesh, const QuadratureRule * quad, double * _op)
    {
        const int n_quad = quad->n;
        const int n_elem = mesh.n_elem();

        auto a = reshape(A, n_var, n_var);
        auto op = reshape(_op, n_var, n_var, n_quad, n_elem);

        auto w = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    for (int c = 0; c < n_var; ++c)
                    {
                        op(c, d, i, el) = a(d, c) * w(i);
                    }
                }
            }
        }
    }

    static void op_1d(int n_var, const double * A, const Mesh1D& mesh, const QuadratureRule * quad, double * _op, const dmat& Pt)
    {
        const int n_elem = mesh.n_elem();
        const int n_quad = quad->n;
        const int n_colloc = Pt.shape()[0];

        auto a = reshape(A, n_var, n_var, n_colloc, n_elem);
        auto op = reshape(_op, n_var, n_var, n_quad, n_elem);

        auto w = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    for (int c = 0; c < n_var; ++c)
                    {
                        double aq = 0.0;
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            aq += Pt(k, i) * a(d, c, k, el);
                        }
                        op(c, d, i, el) = aq * w(i);
                    }
                }
            }
        }
    }

    template <>
    Div<false>::Div(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const double * A, bool constant_coefficient, const QuadratureRule * quad)
        : dim(1),
          n_var(nvar),
          n_colloc(basis->n),
          n_elem(mesh.n_elem())
    {
        if (quad == nullptr)
        {
            int p = (n_colloc - 2) + (n_colloc - 1);
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

        _op.reshape(n_var * n_var * n_quad * n_elem);

        Uq.reshape(n_var * n_quad);
        Fq.reshape(n_var * n_quad);
        
        Df.reshape(n_quad * n_var);
        Pg.reshape(n_quad * n_var);

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

        if (constant_coefficient)
            op_1d_cc(n_var, A, mesh, quad, _op);
        else
            op_1d(n_var, A, mesh, quad, _op, Pt);
    }

    static void div_2d_q(int n_elem,
                         int n_colloc,
                         int n_quad,
                         int n_var,
                         const double * u_,
                         double * du_,
                         const dmat& D,
                         const dmat& Dt,
                         const dmat& P,
                         const dmat& Pt,
                         const double * _op,
                         double * _Uq,
                         double * _Fq,
                         double * _Df,
                         double * _Pg)
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_colloc, n_elem);

        auto op = reshape(_op, 2, n_var, n_var, n_quad, n_quad, n_elem);
        auto Uq = reshape(_Uq, n_var, n_quad, n_quad);
        auto Fq = reshape(_Fq, 2, n_var, n_quad, n_quad);
        auto Df = reshape(_Df, n_quad, n_var, n_colloc);
        auto Pu = reshape(_Df, n_colloc, n_var, n_quad);
        auto Pg = reshape(_Pg, n_quad, n_var, n_colloc);

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

    static void div_1d_q(const int n_elem,
                         const int n_colloc,
                         const int n_quad,
                         const int n_var,
                         const double * u_,
                         double * du_,
                         const dmat& D,
                         const dmat& Dt,
                         const dmat& P,
                         const dmat& Pt,
                         const double * _op,
                         double * _Uq,
                         double * _Fq,
                         double * _Df,
                         double * _Pg)
    {
        auto u = reshape(u_, n_var, n_colloc, n_elem);
        auto du = reshape(du_, n_var, n_colloc, n_elem);

        auto op = reshape(_op, n_var, n_var, n_quad, n_elem);

        auto Uq = reshape(_Uq, n_var);
        auto Fq = reshape(_Fq, n_quad, n_var);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                // evaluate u on quadrature point
                for (int d = 0; d < n_var; ++d)
                {
                    double uq = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        uq += Pt(k, i) * u(d, k, el);
                    }
                    Uq(d) = uq;
                }

                // evaluate flux
                for (int d = 0; d < n_var; ++d)
                {
                    double Fd = 0.0;
                    for (int c = 0; c < n_var; ++c)
                    {
                        Fd += op(c, d, i, el) * Uq(c);
                    }
                    Fq(i, d) = Fd;
                }
            }

            // integral (F, dv/dx)
            for (int k = 0; k < n_colloc; ++k)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double divF = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        divF += D(i, k) * Fq(i, d);
                    }
                    du(d, k, el) += divF;
                }
            }
        }
    }

    template <>
    void Div<false>::action(const double * u_, double * du_) const
    {
        switch (dim)
        {
        case 1:
            div_1d_q(n_elem, n_colloc, n_quad, n_var, u_, du_, D, Dt, P, Pt, _op, Uq, Fq, Df, Pg);
            break;
        case 2:
            div_2d_q(n_elem, n_colloc, n_quad, n_var, u_, du_, D, Dt, P, Pt, _op, Uq, Fq, Df, Pg);
            break;
        default:
            wdg_error("Div<false>::action error: specified dimension not implemented.");
            break;
        }
    }

} // namespace dg
