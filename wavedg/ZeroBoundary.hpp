#ifndef WDG_ZERO_BOUNDARY_HPP
#define WDG_ZERO_BOUNDARY_HPP

#include <map>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Operator.hpp"
#include "Mesh2D.hpp"

namespace dg
{
    class ZeroBoundary
    {
    public:
        /// @brief initialize ZeroBoundary by specifying boundary faces for
        /// which the degrees of freedom should be zero.
        /// @param mesh the mesh
        /// @param basis the basis set. Only defined for Gauss-Lobatto.
        /// @param n_faces number of boundary faces on which DOFs should be zero.
        /// @param faces length n_faces. faces[i] a the boundary face index on
        /// which the DOFs should be zero.
        ZeroBoundary(const Mesh2D& mesh, const QuadratureRule * basis, int n_faces, const int * faces);

        /// @brief zeros out the boundary values on the specified faces. The
        /// action is performed inplace on u.
        /// @param n_var vector dimension of u
        /// @param u FEMVector
        void action(int n_var, double * u) const;

        /// @brief zeros out the boundary values on the specified faces. The
        /// action is performed inplace on u.
        /// @param u FEMVector
        void action(double * u) const
        {
            return action(1, u);
        }

    private:
        int n_dof;
        ivec I;
    };
} // namespace dg


#endif
