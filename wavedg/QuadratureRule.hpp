#ifndef DG_QUADRATURE_RULE_HPP
#define DG_QUADRATURE_RULE_HPP

#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <memory>

#include "wdg_config.hpp"
#include "jacobi.hpp"

namespace dg
{    
    /// @brief quadrature rule on the interval [-1, 1]
    struct QuadratureRule
    {
    private:
        std::unique_ptr<double[]> _x;
        std::unique_ptr<double[]> _w;

    public:
        enum QuadratureType
        {
            GaussLegendre,
            GaussLobatto,
            Undefined
        };

        const int n; ///< size of quadrature rule, i.e. number of collocation points.
        const double * const x; ///< collocation points of quadrature rule. Has length n.
        const double * const w; ///< quadrature weights. Has length n
        const QuadratureType type; ///< specifies the kind of quadrature rule

        /// @brief copy
        QuadratureRule(const QuadratureRule& q);

        /// @brief move
        QuadratureRule(QuadratureRule&& q);

        /// @brief computes the Gauss-Legendre quadrature rule with n nodes (and n weights).
        ///
        /// Rules for n = 1, ... 10 are tabulated, higher order rules are computed using
        /// Golub-Welsch algorithm for finding roots of orthogonal polynomials, and
        /// refined using Newton's method. These nodes are the roots of the n-th Legendre
        /// polynomial. @n
        /// see: @n
        /// Gene H. Golub, and John H. Welsch. 1969. “Calculation of Gauss QuadratureRule
        /// Rules.” Mathematics of Computation 23 (106): 221-s10. doi:10.2307/2004418.
        /// @param[in] n number of quadrature points
        /// @return the quadrature rule
        static QuadratureRule gauss_legendre(int n);

        /// @brief computes the Gauss-Lobatto quadrature rule with n nodes (and n weights).
        ///
        /// Since the Gauss-Lobatto rule includes the end-points of the
        /// interval, n should be greater than or equal to 2. Rules for n = 2,
        /// ... 10 are tabulated, higher order rules are computed using
        /// Golub-Welsch algorithm for finding roots of orthogonal polynomials,
        /// and refined using Newton's method. These nodes are the extrema of the
        /// n-th Legendre polynomial. @n
        /// see: @n
        /// Gene H. Golub, and John H. Welsch. 1969. “Calculation of Gauss QuadratureRule
        /// Rules.” Mathematics of Computation 23 (106): 221-s10. doi:10.2307/2004418.
        /// @param[in] n number of quadrature points
        /// @return the quadrature rule
        static QuadratureRule gauss_lobatto(int n);

        /// @brief returns a reference to a quadrature rule. This function maintains
        /// a global collection of quadrature rules.
        /// @param[in] n size of quadrature rule.
        /// @param[in] rule type of quadrature rule.
        /// @return reference to quadrature rule. If the `quadrature_rule` is
        /// called with the same arguments, then the same pointer is returned.
        static const QuadratureRule * quadrature_rule(int n, QuadratureType rule = QuadratureRule::GaussLegendre);

    private:
        explicit QuadratureRule(int, QuadratureType);
    };
} // namespace dg

#endif