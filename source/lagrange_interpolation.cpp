#include "lagrange_interpolation.hpp"

namespace dg
{
    LagrangeInterpolator::LagrangeInterpolator(int n_, const double * x_) : n(n_), w(n), x(x_)
    {
        if (n < 1)
            throw std::logic_error("cannot initialize with LagrangeInterpolator with n < 1");
        if (x_ == nullptr)
            throw std::runtime_error("cannot initialize LagrangeInterpolator with x == nullptr");
        barycentric_weights(w.data(), x, n);
    }

    void lagrange_basis(double * B_, int ngrid, const double * xgrid, int neval, const double * xeval)
    {
        LagrangeInterpolator I(ngrid, xgrid);

        std::vector<double> y(ngrid, 0.0);

        auto B = reshape(B_, neval, ngrid);

        for (int j=0; j < ngrid; ++j)
        {
            y[j] = 1.0;
            for (int i=0; i < neval; ++i)
            {
                B(i, j) = I.interp(xeval[i], y.data());
            }
            y[j] = 0.0;
        }
    }

    void lagrange_basis_deriv(double * D_, int ngrid, const double * xgrid, int neval, const double * xeval)
    {
        LagrangeInterpolator I(ngrid, xgrid);

        std::vector<double> y(neval, 0.0);

        auto D = reshape(D_, neval, ngrid);

        for (int j=0; j < ngrid; ++j)
        {
            y[j] = 1.0;
            for (int i = 0; i < neval; ++i)
            {
                D(i, j) = I.deriv(xeval[i], y.data());
            }
            y[j] = 0.0;
        }
    }

    void barycentric_weights(double * w, const double * x, int n)
    {
        std::fill_n(w, n, 1.0);

        for (int i=0; i < n; ++i)
        {
            for (int j=0; j < n; ++j)
            {
                if (i == j)
                    continue;
                
                w[i] *= x[i] - x[j];
            }
            w[i] = 1.0 / w[i];
        }

        auto [wmin, wmax] = std::minmax_element(w, w+n);
        double diff = (*wmax) - (*wmin);
        for (int i=0; i < n; ++i)
            w[i] /= diff;
    }

    double lagrange_interpolation(double t, const double * x, const double * w, const double * y, int n)
    {
        double A = 0.0;
        double B = 0.0;

        constexpr double eps = std::numeric_limits<double>::epsilon();

        for (int i=0; i < n; ++i)
        {
            double xdiff = t - x[i];

            if (t == x[i] || std::abs(xdiff) <= eps)
                return y[i];

            double C = w[i] / xdiff;
            A += C * y[i];
            B += C;
        }

        return A / B;
    }

    double lagrange_derivative(double x0, const double * x, const double * w, const double * y, int n)
    {
        bool atnode = false;
        int i;
        
        double A = 0.0;
        double B = 0.0;

        double p = lagrange_interpolation(x0, x, w, y, n);

        constexpr double eps = std::numeric_limits<double>::epsilon();

        for (int j=0; j < n; ++j)
        {
            if (x0 == x[j] || std::abs(x0 - x[j]) <= eps)
            {
                atnode = true;
                B = -w[j];
                i = j;
            }
        }

        if (atnode)
        {
            for (int j=0; j < n; ++j)
            {
                if (j == i)
                    continue;
                
                A += w[j] * (p - y[j]) / (x0 - x[j]);
            }
        }
        else
        {
            for (int j=0; j < n; ++j)
            {
                double t = w[j] / (x0 - x[j]);
                A += t * (p - y[j]) / (x0 - x[j]);
                B += t;
            }
        }

        return A / B;
    }
} // namespace dg