#ifndef WDG_JACOBI_P_HPP
#define DG_JACOBI_HPP

#include <cmath>

#include "config.hpp"

namespace dg
{
    // uses the three-term recurrence relationship to compute the m-th order jacobi
    // polynomial with parameters (a,b) at x in terms of the (m-1)-th order
    // polynomial, y1, and the (m-2)-th order polynomial, y2.
    double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2);

    // evaluates the n-th order jacobi polynomial with parameters (a,b) at x.
    double jacobiP(unsigned int n, double a, double b, double x);

    // evaluates the k-th derivative of the n-th order jacobi polynomial with
    // parameters (a,b) at x.
    double jacobiP_derivative(unsigned int k, unsigned int n, double a, double b, double x);
} // namespace dg

#endif