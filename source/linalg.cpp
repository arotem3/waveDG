#include "linalg.hpp"

namespace dg
{
    bool chol(int m, double * a_)
    {
        auto a = reshape(a_, m, m);
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

    void solve_chol(int m, const double * a_, int n, double * x_)
    {
        auto a = reshape(a_, m, m);
        auto x = reshape(x_, n, m);

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

    void mult_chol(int m, const double * a_, int n, double * x_)
    {
        auto a = reshape(a_, m, m);
        auto x = reshape(x_, n, m);

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
} // namespace dg
