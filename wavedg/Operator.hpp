#ifndef WDG_OPERATOR_HPP
#define WDG_OPERATOR_HPP

#include "wdg_config.hpp"

namespace dg
{
    /// @brief abstract representation of an operator A equipped with an action(x, y) such that y <- A*x.
    class Operator
    {
    public:
        /// @brief y <- A * x
        /// @param[in] x
        /// @param[in,out] y On exit, y <- A * x
        virtual void action(const double * x, double * y) const = 0;
    };

    /// @brief abstract representation of operator that is also invertible, that is inv(x): x <- A \ x.
    class InvertibleOperator : public Operator
    {
    public:
        /// @brief x <- A \ x inplace
        /// @param[in,out] x On exit, x <- A \ x
        virtual void inv(double * x) const = 0;
    };
} // namespace dg


#endif