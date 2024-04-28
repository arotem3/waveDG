#ifndef WDG_MASK_HPP
#define WDG_MASK_HPP

#include <map>

#include "wdg_config.hpp"
#include "Mesh2D.hpp"
#include "QuadratureRule.hpp"

namespace dg
{
    class CGMask
    {
    public:
        CGMask(const Mesh2D& mesh, const QuadratureRule * basis);

        /// @brief zeros out redundant degrees of freedom of the DG vector u representing a CG vector.
        void mask(int n_var, double * u) const;

        /// @brief sets redundant degrees of freedom of the DG vector u to their continuous values.
        void unmask(int n_var, double * u) const;

        /// @brief sums over all elements for each DOF
        void sum(int n_var, double * u) const;
    
    private:
        const int n_dof;
        imat I;
    };
} // namespace dg


#endif
