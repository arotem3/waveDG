#ifndef DG_QUADRATURE_RULE_HPP
#define DG_QUADRATURE_RULE_HPP

#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <memory>

#include "config.hpp"
#include "jacobi.hpp"

namespace dg
{    
    enum class QuadratureType
    {
        GaussLegendre,
        GaussLobatto,
        Undefined
    };

    struct QuadratureRule
    {
    private:
        std::unique_ptr<double[]> _x;
        std::unique_ptr<double[]> _w;

    public:
        const int n;
        const double * const x;
        const double * const w;
        const QuadratureType type;

        friend QuadratureRule gauss_legendre(int);
        friend QuadratureRule gauss_lobatto(int);

        explicit QuadratureRule(int, QuadratureType);
        QuadratureRule(const QuadratureRule& q);
        QuadratureRule(QuadratureRule&& q);
    };

    // computes the Gauss-Legendre quadrature rule with n nodes (and n weights).
    // Rules for n = 1, ... 10 are tabulated, higher order rules are computed using
    // Golub-Welsch algorithm for finding roots of orthogonal polynomials, and
    // refined using Newton's method. These nodes are the roots of the n-th Legendre
    // polynomial.
    // see:
    // Gene H. Golub, and John H. Welsch. 1969. “Calculation of Gauss QuadratureRule
    // Rules.” Mathematics of Computation 23 (106): 221-s10. doi:10.2307/2004418.
    QuadratureRule gauss_legendre(int n);

    // computes the Gauss-Lobatto quadrature rule with n nodes (and n weights).
    // Rules for n = 1, ... 10 are tabulated, higher order rules are computed using
    // Golub-Welsch algorithm for finding roots of orthogonal polynomials, and
    // refined using Newton's method. These nodes are the roots of the n-th Legendre
    // polynomial.
    // see:
    // Gene H. Golub, and John H. Welsch. 1969. “Calculation of Gauss QuadratureRule
    // Rules.” Mathematics of Computation 23 (106): 221-s10. doi:10.2307/2004418.
    QuadratureRule gauss_lobatto(int n);

    /// @brief returns a reference to a quadrature rule. This function maintains
    /// a global collection of quadrature rules.
    /// @param n size of quadrature rule.
    /// @param rule type of quadrature rule.
    /// @return reference to quadrature rule.
    const QuadratureRule * quadrature_rule(int n, QuadratureType rule = QuadratureType::GaussLegendre);
} // namespace dg

#endif