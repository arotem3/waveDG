#include "MassMatrix.hpp"

static inline double square(double x)
{
    return x * x;
}

// computes the (lower) Cholesky decomposition of a inplace. 
static bool chol(int m, double * a_)
{
    auto a = dg::reshape(a_, m, m);
    for (int k = 0; k < m; ++k)
    {
        for (int j = k+1; j < m; ++j)
        {
            const double s = a(j, k) / a(k, k);
            for (int i = j; i < m; ++i)
            {
                a(i, j) -= s * a(i, k);
            }
        }

        if (a(k, k) == 0)
            return false;

        double s = 1.0 / std::sqrt(a(k, k));
        for (int i = k; i < m; ++i)
        {
            a(i, k) *= s;
        }
    }

    return true;
}

// solve A\x assuming a stores the lower Cholesky factor of a
static void solve_chol(int m, const double * a_, int n, double * x_)
{
    auto a = dg::reshape(a_, m, m);
    auto x = dg::reshape(x_, n, m);

    // solve L\x
    for (int j = 0; j < m; ++j)
    {
        for (int d = 0; d < n; ++d)
            x(d, j) /= a(j, j);
        
        for (int i = j+1; i < m; ++i)
        {
            for (int d = 0; d < n; ++d)
            {
                x(d, i) -= a(i, j) * x(d, j);
            }
        }
    }

    // solve L'\x
    for (int j = m-1; j >= 0; --j)
    {
        for (int d = 0; d < n; ++d)
            x(d, j) /= a(j, j);

        for (int i = 0; i < j; ++i)
        {
            for (int d = 0; d < n; ++d)
            {
                x(d, i) -= a(j, i) * x(d, j);
            }
        }
    }
}

// multiplies A*x assuming a stores the lower Cholesky factor of a
static void mult_chol(int m, const double * a_, int n, double * x_)
{
    auto a = dg::reshape(a_, m, m);
    auto x = dg::reshape(x_, n, m);

    // L' * x
    for (int i = 0; i < m; ++i)
    {
        for (int d = 0; d < n; ++d)
        {
            double y = 0.0;
            for (int j = i; j < m; ++j)
            {
                y += a(j, i) * x(d, j);
            }
            x(d, i) = y;
        }
    }

    // L * x
    for (int i = m-1; i >= 0; --i)
    {
        for (int d = 0; d < n; ++d)
        {
            double y = 0.0;
            for (int j = 0; j <= i; ++j)
            {
                y += a(i, j) * x(d, j);
            }
            x(d, i) = y;
        }
    }
}

namespace dg
{
    template <>
    MassMatrix<true>::MassMatrix(const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()), n_colloc(basis->n)
    {
        const double * _detJ = mesh.element_measures(basis);
        auto detJ = reshape(_detJ, n_colloc, n_colloc, n_elem);

        m.resize(n_colloc * n_colloc * n_elem);
        auto M = reshape(m.data(), n_colloc, n_colloc, n_elem);
        for (int el = 0; el < n_elem; ++el)
        {
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    M(i, j, el) = basis->w[i] * basis->w[j] * detJ(i, j, el);
                }
            }
        }
    }

    template <>
    void MassMatrix<true>::operator()(const double * x_, double * y_, int n_var) const
    {
        const int n = n_colloc * n_colloc * n_elem;
        auto x = reshape(x_, n_var, n);
        auto y = reshape(y_, n_var, n);

        for (int i=0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                y(d, i) = m[i] * x(d, i);
            }
        }
    }

    template <>
    void MassMatrix<true>::inv(double * x_, int n_var) const
    {
        const int n = n_colloc * n_colloc * n_elem;
        auto x = reshape(x_, n_var, n);

        for (int i=0; i < n; ++i)
        {
            for (int d = 0; d < n_var; ++d)
            {
                x(d, i) /= m[i];
            }
        }
    }

    template <>
    MassMatrix<false>::MassMatrix(const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_elem(mesh.n_elem()), n_colloc(basis->n)
    {
        if (quad == nullptr)
            throw std::runtime_error("cannot pass nullptr for the quadrature rule for MassMatrix<Diagonal> with Diagonal = false.");
            
        const int n_quad = quad->n;
        const int q2d = n_quad * n_quad;
        const int c2d = n_colloc * n_colloc;

        const double * _detJ = mesh.element_measures(quad);
        auto detJ = reshape(_detJ, q2d, n_elem);

        dmat B(n_quad, n_colloc);
        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);

        m.resize(c2d * c2d * n_elem, 0.0);
        auto M = reshape(m.data(), c2d, c2d, n_elem);

        for (int el = 0; el < n_elem; ++el)
        {
            // eval element mass matrix
            for (int j = 0; j < c2d; ++j)
            {
                for (int i = 0; i < c2d; ++i)
                {
                    double mij = 0.0;
                    for (int p = 0; p < q2d; ++p)
                    {
                        const double bi = B(p%n_quad, i%n_colloc) * B(p/n_quad, i/n_colloc);
                        const double bj = B(p%n_quad, j%n_colloc) * B(p/n_quad, j/n_colloc);
                        const double w = detJ(p, el) * quad->w[p%n_quad] * quad->w[p/n_quad];
                        mij += bi * bj * w;
                    }
                    M(i, j, el) = mij;
                }
            }

            // cholesky
            const bool successful_factorization = chol(c2d, &M(0, 0, el));

            if (not successful_factorization)
            {
                throw std::runtime_error("Failed to compute Cholesky decomposition of mass matrix");
            }
        }
    }

    template <>
    void MassMatrix<false>::operator()(const double * x_, double * y_, int n_var) const
    {
        const int c2d = n_colloc * n_colloc;
        const int block = c2d * c2d;

        for (int el = 0; el < n_elem; ++el)
        {
            const double * M = m.data() + block * el;
            
            const int offset = n_var * c2d * el;
            const double * x = x_ + offset;
            double * y = y_ + offset;

            for (int i=0; i < n_var*c2d; ++i)
            {
                y[i] = x[i];
            }

            mult_chol(c2d, M, n_var, y);
        }
    }

    template <>
    void MassMatrix<false>::inv(double * x_, int n_var) const
    {
        const int c2d = n_colloc * n_colloc;
        const int block = c2d * c2d;

        for (int el = 0; el < n_elem; ++el)
        {
            const double * M = m.data() + block * el;
            
            const int offset = n_var * c2d * el;
            double * x = x_ + offset;

            solve_chol(c2d, M, n_var, x);
        }
    }
    
    template <>
    EdgeMassMatrix<true>::EdgeMassMatrix(const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_edges(mesh.n_edges(edge_type)), n_colloc(basis->n), m(n_colloc*n_edges)
    {
        const double * _ds = mesh.edge_measures(basis, edge_type);
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
    void EdgeMassMatrix<true>::operator()(const double * x_, double * y_, int n_var) const
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
    EdgeMassMatrix<false>::EdgeMassMatrix(const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule* basis, const QuadratureRule* quad)
        : n_edges(mesh.n_edges(edge_type)), n_colloc(basis->n), m(n_colloc*n_colloc*n_edges)
    {
        if (quad == nullptr)
            throw std::runtime_error("cannot pass nullptr for the quadrature rule for EdgeMassMatrix<Diagonal> with Diagonal = false.");

        const int n_quad = quad->n;

        const double * _ds = mesh.edge_measures(quad, edge_type);
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
                throw std::runtime_error("Failed to compute Cholesky decomposition of mass matrix");
        }
    }

    template <>
    void EdgeMassMatrix<false>::operator()(const double * x_, double * y_, int n_var) const
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
