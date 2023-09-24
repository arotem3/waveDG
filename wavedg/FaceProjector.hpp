#ifndef DG_FACE_PROJECTOR_HPP
#define DG_FACE_PROJECTOR_HPP

#include "config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    // projects grid functions to element faces
    class FaceProjector
    {
    private:
        const int n_elem;
        const int n_edges;
        const int n_colloc;
        QuadratureType basis_type;
        EdgeType edge_type;
        const double * _n;
        std::vector<double> V;
        std::vector<int> vol_to_edge;

    public:
        FaceProjector(const Mesh2D& mesh, const QuadratureRule * basis, EdgeType edge_type_);

        /// @brief Prolongs element values to faces
        /// @param x shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param y shape (n_var, n_colloc, 2, n_edges). Edge values
        /// @param n_var vector dimension of grid function
        void operator()(const double * x, double * y, int n_var=1) const;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param y shape (n_var, n_colloc, 2, n_edges), Edge values
        /// @param x shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, x = x + A' * y where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        void t(const double * y, double * x, int n_var=1) const;

        /// @brief Prolongs element values of vector field grid function
        /// to element faces.
        /// @param x shape (n_var, 2, n_colloc, n_elem) or (n_var, 2) if
        /// constant_value. Element values
        /// @param y shape (n_var, n_colloc, 2, n_edges). Edge values of n.x
        /// @param constant_value specifies whether x is constant throughout the
        /// domain. This is applicable when computing the normal component a
        /// constant velocity coefficient.
        /// @param n_var vector dimension of grid function
        void project_normal(const double * x, double * y, bool constant_value, int n_var=1) const;
    };
} // namespace dg


#endif