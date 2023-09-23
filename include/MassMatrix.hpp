#ifndef DG_MASS_MATRIX_HPP
#define DG_MASS_MATRIX_HPP

#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    template <bool Diagonal>
    class MassMatrix
    {
    private:
        const int n_elem;
        const int n_colloc;
        std::vector<double> m;

    public:
        MassMatrix(const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);

        /// @brief y = a * M * x
        /// @param x shape (n_var, n, n, n_elem) where n is the size of the 1D
        /// basis set specified on initialization. The DG grid function.
        /// @param y shape (n_var, n, n, n_elem). On exit, y <- a * M * x
        /// @param n_var vector dimension of x and y
        void operator()(const double * x, double * y, int n_var = 1) const;

        /// @brief x = M \ x
        /// @param x shape (n_var, n, n, n_elem), where n is the size of the 1D
        /// basis set specified on initialization. The DG grid function. On
        /// exit, x <- M \ x.
        /// @param n_var vector dimension of x
        void inv(double * x, int n_var=1) const;
    };

    template <bool Diagonal>
    class EdgeMassMatrix
    {
    private:
        const int n_edges;
        const int n_colloc;
        std::vector<double> m;
    
    public:
        EdgeMassMatrix(const Mesh2D& mesh, EdgeType edge_type, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);

        void operator()(const double * x, double * y, int n_var=1) const;

        void inv(double * x, int n_var=1) const;
    };
} // namespace dg

#endif