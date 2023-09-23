#ifndef DG_LAGRANGE_INTERPOLATION_HPP
#define DG_LAGRANGE_INTERPOLATION_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

#include "Tensor.hpp"

namespace dg
{
    /// @brief computes the Vandermonde matrix for the Lagrange basis functions
    /// defined on xgrid evaluated on the points xeval.
    /// @param B size (neval, ngrid); on exit, is the Vandermonde matrix such that B(i, j) = basis[j](x[i])
    /// @param ngrid number of grid points/basis functions
    /// @param xgrid the grid points that the basis functions collocate
    /// @param neval number of points where the basis functions will be evaluated
    /// @param xeval the points to evaluate the basis functions
    void lagrange_basis(double * B, int ngrid, const double * xgrid, int neval, const double * xeval);

    /// @brief computes the Vandermonde matrix for the derivatives of the Lagrange basis functions
    /// defined on xgrid evaluated on the points xeval.
    /// @param D size (neval, ngrid); on exit, is the Vandermonde matrix such that D(i, j) = d/dx basis[j](x[i])
    /// @param ngrid number of grid points/basis functions
    /// @param xgrid the grid points that the basis functions collocate
    /// @param neval number of points where the basis functions will be evaluated
    /// @param xeval the points to evaluate the basis functions
    void lagrange_basis_deriv(double * D, int ngrid, const double * xgrid, int neval, const double * xeval);

    // compute barycentric weights for Lagrange interpolation on the points in x.
    void barycentric_weights(double * w, const double * x, int n);

    // interpolate (x, y) at x0 with precomputed barycentric weights w.
    double lagrange_interpolation(double x0, const double * x, const double * w, const double * y, int n);

    // evaluate the derivative of the polynomial interpolating (x, y) at x0 with precomputed barycentric weights w.
    double lagrange_derivative(double x0, const double * x, const double * w, const double * y, int n);

    class LagrangeInterpolator
    {
    private:
        const int n;
        std::vector<double> w;
        const double * x;

    public:
        LagrangeInterpolator(int n_, const double * x_);

        // interpolate y at x0
        double interp(double x0, const double * y) const
        {
            return lagrange_interpolation(x0, x, w.data(), y, n);
        }

        // evaluate the derivative of the polynomial interpolating (x, y)
        double deriv(double x0, const double * y) const
        {
            return lagrange_derivative(x0, x, w.data(), y, n);
        }
    };
} // namespace dg


#endif