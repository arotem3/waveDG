#ifndef WDG_OPERATOR_HPP
#define WDG_OPERATOR_HPP

#include "wdg_config.hpp"
#include "wdg_error.hpp"

namespace dg
{
    /// @brief abstract representation of an operator A equipped with an action(x, y) such that y <- A*x.
    class Operator
    {
    public:
        Operator() = default;
        virtual ~Operator() = default;

        /// @brief y <- A * x
        /// @param[in] x
        /// @param[in,out] y On exit, y <- A * x
        virtual void action(const double * x, double * y) const = 0;

        /// @brief y <- A * x
        /// @param[in] n_var vector dimension of x and y
        /// @param[in] x 
        /// @param[in,out] y On exit, y <- A * x
        virtual void action(int n_var, const double * x, double * y) const = 0;
    };

    /// @brief abstract representation of operator that is also invertible, that is inv(x): x <- A \ x.
    class InvertibleOperator : public Operator
    {
    public:
        InvertibleOperator() = default;
        virtual ~InvertibleOperator() = default;

        /// @brief x <- A \ x inplace
        /// @param[in,out] x On exit, x <- A \ x
        virtual void inv(double * x) const = 0;

        /// @brief x <- A \ x inplace
        /// @param[in] n_var vector dimension of x
        /// @param[in,out] x On exit, x <- A \ x
        virtual void inv(int n_var, double * x) const = 0;
    };
} // namespace dg


#endif