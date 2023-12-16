#ifndef DG_JACOBI_HPP
#define DG_JACOBI_HPP

#include <cmath>

#include "wdg_config.hpp"

namespace dg
{
    /// @brief uses the three-term recurrence relationship to compute the Jacobi
    /// polynomial \f$P_m^{(a,b)}(x)\f$ in terms of \f$y_1=P_{m-1}^{(a,b)}(x)\f$ and \f$y_2=P_{m-2}^{(a,b)}(x)\f$.
    /// @param m order of Jacobi polynomial
    /// @param a Parameter of Jacobi polynomial
    /// @param b Parameter of Jacobi polynomial
    /// @param x point to evaluate \f$P_m^{(a,b)}(x)\f$.
    /// @param y1 \f$y_1=P_{m-1}^{(a,b)}(x)\f$
    /// @param y2 \f$y_2=P_{m-2}^{(a,b)}(x)\f$
    /// @return 
    double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2);

    /// @brief evaluates the n-th order jacobi polynomial with parameters (a,b) at x.
    ///
    /// uses the recurence relationship defined by `JacobiP_next`.
    double jacobiP(unsigned int n, double a, double b, double x);

    /// @brief evaluates the k-th derivative of the n-th order jacobi polynomial with
    /// parameters (a,b) at x. That is \f$\frac{d^k}{d x^k}P_n^{(a,b)}(x)\f$
    double jacobiP_derivative(unsigned int k, unsigned int n, double a, double b, double x);
} // namespace dg

#endif