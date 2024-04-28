#ifndef WDG_DG2GC_HPP
#define WDG_DG2GC_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Operator.hpp"
#include "Mesh2D.hpp"
#include "CGMask.hpp"
#include "linalg.hpp"

namespace dg
{
    /// @brief Makes a DG function continuous. Only for Gauss-Lobatto basis functions.
    class DG2CG
    {
    public:
        DG2CG(const CGMask& m, const Mesh2D& mesh, const QuadratureRule * basis);

        /// @brief project u to continuous Galerkin basis. Projection is performed inplace on u.
        void action(int nvar, double * u) const;

        /// @brief project u to continuous Galerkin basis. Projection is performed inplace on u.
        void action(double * u) const
        {
            action(1, u);
        }

    private:
        const int n_dof;
        const CGMask& cgm;
        dvec m;
        dvec inv_m;
    };
} // namespace dg

#endif
