#include "MassMatrix.hpp"

namespace dg
{
// MassMatrix<true>
    template <>
    MassMatrix<true>::MassMatrix(int nv, const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          n_var(nv),
          m(n_colloc * n_colloc * n_elem)
    {
        const double * _detJ = mesh.element_metrics(basis).measures();
        auto detJ = reshape(_detJ, n_colloc, n_colloc, n_elem);
        auto w = reshape(basis->w, basis->n);

        auto M = reshape(m.data(), n_colloc, n_colloc, n_elem);
        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    M(i, j, el) = w(i) * w(j) * detJ(i, j, el);
                }
            }
        }
    }

    template <>
    void MassMatrix<true>::action(const double * x_, double * y_) const
    {
        const int n = n_colloc * n_colloc * n_elem;
        auto x = reshape(x_, n_var, n);
        auto y = reshape(y_, n_var, n);

        for (int i=0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                y(d, i) = m(i) * x(d, i);
            }
        }
    }

    template <>
    void MassMatrix<true>::inv(double * x_) const
    {
        const int n = n_colloc * n_colloc * n_elem;
        auto x = reshape(x_, n_var, n);

        for (int i=0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                x(d, i) /= m(i);
            }
        }
    }

// MassMatrix<false>
    template <>
    MassMatrix<false>::MassMatrix(int nv, const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          n_var(nv),
          m(n_colloc * n_colloc * n_colloc * n_colloc * n_elem)
    {
        if (quad == nullptr)
        {
            int p = 2 * (n_colloc - 1) + 2*mesh.max_element_order();
            p = 1 + p/2;

            quad = QuadratureRule::quadrature_rule(p);
        }

        const int n_quad = quad->n;
        const int q2d = n_quad * n_quad;
        const int c2d = n_colloc * n_colloc;

        const double * _detJ = mesh.element_metrics(quad).measures();
        auto detJ = reshape(_detJ, q2d, n_elem);

        auto W = reshape(quad->w, quad->n);

        dmat B(n_quad, n_colloc);
        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);

        auto M = reshape(m.data(), c2d, c2d, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            // eval element mass matrix
            for (int j = 0; j < c2d; ++j)
            {
                for (int i = j; i < c2d; ++i) // only lower triangular part
                {
                    double mij = 0.0;
                    for (int p = 0; p < q2d; ++p)
                    {
                        const double bi = B(p%n_quad, i%n_colloc) * B(p/n_quad, i/n_colloc);
                        const double bj = B(p%n_quad, j%n_colloc) * B(p/n_quad, j/n_colloc);
                        const double w = detJ(p, el) * W(p%n_quad) * W(p/n_quad);
                        mij += bi * bj * w;
                    }
                    M(i, j, el) = mij;
                }
            }

            // cholesky
            const bool successful_factorization = chol(c2d, &M(0, 0, el));

            if (not successful_factorization)
            {
                wdg_error("MassMatrix<false> error: Failed to compute Cholesky decomposition of mass matrix.");
            }
        }
    }

    template <>
    void MassMatrix<false>::action(const double * x_, double * y_) const
    {
        const int c2d = n_colloc * n_colloc;
        const int block = c2d * c2d;

        auto x = reshape(x_, n_var * c2d, n_elem);
        auto y = reshape(y_, n_var * c2d, n_elem);
        auto M = reshape(m.data(), block, n_elem);
        
        for (int el = 0; el < n_elem; ++el)
        {
            for (int i=0; i < n_var * c2d; ++i)
            {
                y(i, el) = x(i, el);
            }

            mult_chol(c2d, &M(0, el), n_var, &y(0, el));
        }
    }

    template <>
    void MassMatrix<false>::inv(double * x_) const
    {
        const int c2d = n_colloc * n_colloc;
        const int block = c2d * c2d;

        auto x = reshape(x_, n_var * c2d, n_elem);
        auto M = reshape(m.data(), block, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            solve_chol(c2d, &M(0, el), n_var, &x(0, el));
        }
    }

// WeightedMassMatrix<true>
    template <>
    WeightedMassMatrix<true>::WeightedMassMatrix(int nv, const Mesh2D& mesh, const double * A_, bool A_is_diagonal, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          n_var(nv),
          diag_coef(A_is_diagonal)
    {
        const double * _detJ = mesh.element_metrics(basis).measures();
        auto detJ = reshape(_detJ, n_colloc, n_colloc, n_elem);
        auto w = reshape(basis->w, basis->n);

        if (diag_coef)
        {
            m.reshape(n_var * n_colloc * n_colloc * n_elem);
            auto M = reshape(m.data(), n_var, n_colloc, n_colloc, n_elem);
            auto A = reshape(A_, n_var, n_colloc, n_colloc, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int j = 0; j < n_colloc; ++j)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            const double a = A(d, i, j, el);

                            if (a <= 0)
                            {
                                wdg_error("WeightedMassMatrix<true> error: Mass matrix weight must be positive definite!");
                            }

                            M(d, i, j, el) = A(d, i, j, el) * w(i) * w(j) * detJ(i, j, el);
                        }
                    }
                }
            }
        }
        else
        {
            const int v2d = n_var * n_var;
            m.reshape(v2d * n_colloc * n_colloc * n_elem);
            auto M = reshape(m.data(), v2d, n_colloc, n_colloc, n_elem);
            auto A = reshape(A_, v2d, n_colloc, n_colloc, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int j = 0; j < n_colloc; ++j)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < v2d; ++d)
                        {
                            M(d, i, j, el) = A(d, i, j, el) * w(i) * w(j) * detJ(i, j, el);
                        }

                        const bool successful_factorization = chol(n_var, &M(0, i, j, el));

                        if (not successful_factorization)
                        {
                            wdg_error("WeightedMassMatrix<true> error: Failed to compute Cholesky decomposition of mass matrix");
                        }
                    }
                }
            }
        }
    }

    template <>
    void WeightedMassMatrix<true>::action(const double * x_, double * y_) const
    {
        if (diag_coef)
        {
            const int n = n_var * n_colloc * n_colloc * n_elem;
            auto x = reshape(x_, n);
            auto y = reshape(y_, n);

            for (int i=0; i < n; ++i)
            {
                y(i) = m(i) * x(i);
            }
        }
        else
        {
            const int v2d = n_var * n_var;
            const int n = n_colloc * n_colloc * n_elem;
            auto x = reshape(x_, n_var, n);
            auto y = reshape(y_, n_var, n);
            auto M = reshape(m.data(), v2d, n);

            for (int i=0; i < n; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    y(d, i) = x(d, i);
                }

                mult_chol(n_var, &M(0, i), 1, &y(0, i));
            }
        }
    }

    template <>
    void WeightedMassMatrix<true>::inv(double * x_) const
    {
        if (diag_coef)
        {
            const int n = n_var * n_colloc * n_colloc * n_elem;
            auto x = reshape(x_, n);

            for (int i=0; i < n; ++i)
            {
                x(i) /= m(i);
            }
        }
        else
        {
            const int v2d = n_var * n_var;
            const int n = n_colloc * n_colloc * n_elem;
            auto x = reshape(x_, n_var, n);
            auto M = reshape(m.data(), v2d, n);

            for (int i=0; i < n; ++i)
            {
                solve_chol(n_var, &M(0, i), 1, &x(0, i));
            }
        }
    }

// WeightedMassMatrix<false>
    template <>
    WeightedMassMatrix<false>::WeightedMassMatrix(int nv, const Mesh2D& mesh, const double * A_, bool A_is_diagonal, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          n_var(nv),
          diag_coef(A_is_diagonal)
    {
        if (quad == nullptr)
        {
            int p = 3*(n_colloc-1) + 2*mesh.max_element_order();
            p = 1 + p/2;

            quad = QuadratureRule::quadrature_rule(p);
        }

        const int n_quad = quad->n;
        const int q2d = n_quad * n_quad;
        const int c2d = n_colloc * n_colloc;

        const double * _detJ = mesh.element_metrics(quad).measures();
        auto detJ = reshape(_detJ, q2d, n_elem);

        auto W = reshape(quad->w, quad->n);

        dmat B(n_quad, n_colloc);
        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);

        if (diag_coef)
        {
            const int block = c2d * c2d;
            m.reshape(n_var * block * n_elem);
            auto M = reshape(m.data(), c2d, c2d, n_var, n_elem);
            auto A = reshape(A_, n_var, c2d, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int j=0; j < c2d; ++j)
                {
                    for (int i=j; i < c2d; ++i)
                    {
                        for (int d=0; d < n_var; ++d)
                        {   
                            // evaluate mass
                            double mij = 0.0;
                            for (int p=0; p < q2d; ++p)
                            {
                                // evaluate A on quadrature points
                                double aq=0.0;
                                for (int k=0; k < c2d; ++k)
                                {
                                    aq += B(p%n_quad, k%n_colloc) * B(p/n_quad, k/n_colloc) * A(d, k, el);
                                }

                                const double bi = B(p%n_quad, i%n_colloc) * B(p/n_quad, i/n_colloc);
                                const double bj = B(p%n_quad, j%n_colloc) * B(p/n_quad, j/n_colloc);
                                const double w = detJ(p, el) * W(p%n_quad) * W(p/n_quad);
                                mij += aq * bi * bj * w;
                            }
                            M(i, j, d, el) = mij;
                        }
                    }
                }

                // compute Cholesky factorization
                for (int d=0; d < n_var; ++d)
                {
                    const bool successful_factorization = chol(c2d, &M(0, 0, d, el));

                    if (not successful_factorization)
                    {
                        wdg_error("WeightedMassMatrix<false> error: Failed to compute Cholesky decomposition of mass matrix.");
                    }
                }
            }
        }
        else
        {
            const int block = c2d * c2d;
            m.reshape(n_var * n_var * block * n_elem);
            auto M = reshape(m.data(), n_var, c2d, n_var, c2d, n_elem);
            auto A = reshape(A_, n_var, n_var, c2d, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int j=0; j < c2d; ++j)
                {
                    for (int i=j; i < c2d; ++i)
                    {
                        for (int d=0; d < n_var; ++d)
                        {
                            for (int k=0; k < n_var; ++k)
                            {
                                // evaluate mass
                                double mij = 0.0;
                                for (int p=0; p < q2d; ++p)
                                {
                                    // evaluate A on quadrature points
                                    double aq=0.0;
                                    for (int l=0; l < c2d; ++l)
                                    {
                                        aq += B(p%n_quad, l%n_colloc) * B(p/n_quad, l/n_colloc) * A(d, k, l, el);
                                    }

                                    const double bi = B(p%n_quad, i%n_colloc) * B(p/n_quad, i/n_colloc);
                                    const double bj = B(p%n_quad, j%n_colloc) * B(p/n_quad, j/n_colloc);
                                    const double w = detJ(p, el) * W(p%n_quad) * W(p/n_quad);
                                    mij += aq * bi * bj * w;
                                }
                                M(d, i, k, j, el) = mij;
                            }
                        }
                    }
                }

                const bool successful_factorization = chol(c2d*n_var, &M(0, 0, 0, 0, el));

                if (not successful_factorization)
                {
                    wdg_error("WeightedMassMatrix<false> error: Failed to compute Cholesky decomposition of mass matrix.");
                }
            }
        }
    }

    template <>
    void WeightedMassMatrix<false>::action(const double * x_, double * y_) const
    {
        if (diag_coef)
        {
            const int c2d = n_colloc * n_colloc;
            const int block = c2d * c2d;
            auto x = reshape(x_, n_var, c2d, n_elem);
            auto y = reshape(y_, n_var, c2d, n_elem);
            auto M = reshape(m.data(), block, n_var, n_elem);
            dvec v(c2d);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int d=0; d < n_var; ++d)
                {
                    for (int i=0; i < n_colloc; ++i)
                    {
                        v(i) = x(d, i, el);
                    }

                    mult_chol(c2d, &M(0, d, el), 1, v.data());

                    for (int i=0; i < n_colloc; ++i)
                    {
                        y(d, i, el) = v(i);
                    }
                }
            }
        }
        else
        {
            const int c2d = n_colloc * n_colloc;
            const int eldof = n_var * c2d;
            const int block = eldof * eldof;
            auto x = reshape(x_, n_var * c2d, n_elem);
            auto y = reshape(y_, n_var * c2d, n_elem);
            auto M = reshape(m.data(), block, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int i=0; i < eldof; ++i)
                {
                    y(i, el) = x(i, el);
                }

                mult_chol(eldof, &M(0, el), 1, &y(0, el));
            }
        }
    }

    template <>
    void WeightedMassMatrix<false>::inv(double * x_) const
    {
        if (diag_coef)
        {
            const int c2d = n_colloc * n_colloc;
            const int block = c2d * c2d;
            auto x = reshape(x_, n_var, c2d, n_elem);
            auto M = reshape(m.data(), block, n_var, n_elem);
            dvec v(c2d);

            for (int el = 0; el < n_elem; ++el)
            {
                for (int d=0; d < n_var; ++d)
                {
                    for (int i=0; i < n_colloc; ++i)
                    {
                        v(i) = x(d, i, el);
                    }

                    solve_chol(c2d, &M(0, d, el), 1, v.data());

                    for (int i=0; i < n_colloc; ++i)
                    {
                        x(d, i, el) = v(i);
                    }
                }
            }
        }
        else
        {
            const int c2d = n_colloc * n_colloc;
            const int eldof = n_var * c2d;
            const int block = eldof * eldof;
            auto x = reshape(x_, n_var * c2d, n_elem);
            auto M = reshape(m.data(), block, n_elem);

            for (int el = 0; el < n_elem; ++el)
            {
                mult_chol(eldof, &M(0, el), 1, &x(0, el));
            }
        }
    }

// EdgeMassMatrix
    template <>
    EdgeMassMatrix<true>::EdgeMassMatrix(const Mesh2D& mesh, Edge::EdgeType edge_type, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_edges(mesh.n_edges(edge_type)), n_colloc(basis->n), m(n_colloc*n_edges)
    {
        const double * _ds = mesh.edge_metrics(basis, edge_type).measures();
        auto ds = reshape(_ds, n_colloc, n_edges);

        auto M = reshape(m.data(), n_colloc, n_edges);
        for (int e = 0; e < n_edges; ++e)
        {
            for (int i = 0; i < n_colloc; ++i)
            {
                M(i, e) = basis->w[i] * ds(i, e);
            }
        }
    }

    template <>
    void EdgeMassMatrix<true>::action(const double * x_, double * y_, int n_var) const
    {
        const int n = n_colloc * n_edges;
        auto x = reshape(x_, n_var, n);
        auto y = reshape(y_, n_var, n);

        for (int i = 0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                y(d, i) = m[i] * x(d, i);
            }
        }
    }

    template <>
    void EdgeMassMatrix<true>::inv(double * x_, int n_var) const
    {
        const int n = n_colloc * n_edges;
        auto x = reshape(x_, n_var, n);

        for (int i = 0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                x(d, i) /= m[i];
            }
        }
    }

    template <>
    EdgeMassMatrix<false>::EdgeMassMatrix(const Mesh2D& mesh, Edge::EdgeType edge_type, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_edges(mesh.n_edges(edge_type)), n_colloc(basis->n), m(n_colloc*n_colloc*n_edges)
    {
        if (quad == nullptr)
            wdg_error("EdgeMassMatrix<false> error: cannot pass nullptr for the quadrature rule for EdgeMassMatrix<Diagonal> with Diagonal = false.");

        const int n_quad = quad->n;

        const double * _ds = mesh.edge_metrics(quad, edge_type).measures();
        auto ds = reshape(_ds, n_quad, n_edges);

        dmat B(n_quad, n_colloc);
        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);

        auto M = reshape(m.data(), n_colloc, n_colloc, n_edges);

        for (int e = 0; e < n_edges; ++e)
        {
            // eval mass matrix
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    double mij = 0.0;
                    for (int p = 0; p < n_quad; ++p)
                    {
                        const double w = quad->w[p] * ds(p, e);
                        mij += B(p, i) * B(p, j) * w;
                    }
                    M(i, j, e) = mij;
                }
            }

            // factor
            const bool successful_factorization = chol(n_colloc, &M(0, 0, e));


            if (not successful_factorization)
                wdg_error("EdgeMassMatrix<false> error: Failed to compute Cholesky decomposition of mass matrix.");
        }
    }

    template <>
    void EdgeMassMatrix<false>::action(const double * x_, double * y_, int n_var) const
    {
        const int block = n_colloc * n_colloc;

        for (int e = 0; e < n_edges; ++e)
        {
            const double * M = m.data() + e * block;

            const int offset = n_var * n_colloc * e;
            const double * x = x_ + offset;
            double * y = y_ + offset;

            for (int i = 0; i < n_var*n_colloc; ++i)
            {
                y[i] = x[i];
            }

            mult_chol(n_colloc, M, n_var, y);
        }
    }

    template <>
    void EdgeMassMatrix<false>::inv(double * x_, int n_var) const
    {
        const int block = n_colloc * n_colloc;
        
        for (int e = 0; e < n_edges; ++e)
        {
            const double * M = m.data() + e * block;

            double * x = x_ + e * n_var * n_colloc;

            solve_chol(n_colloc, M, n_var, x);
        }
    }
} // namespace dg
