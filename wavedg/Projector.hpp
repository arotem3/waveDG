#ifndef DG_PROJECT_HPP
#define DG_PROJECT_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "MassMatrix.hpp"
#include "LinearFunctional.hpp"

namespace dg
{
    /// @brief Projects functions onto basis set.
    ///
    /// @details Finds an \f$\tilde f\in\mathrm{span}\{\phi\}\f$ such that for every basis function \f$\phi\f$
    /// $$(\tilde{f}, \phi) = (f, \phi).$$
    class Projector
    {
    private:
        LinearFunctional lf;
        const InvertibleOperator * m;

    public:
        /// @brief initialize projector
        /// @tparam Diagonal template parameter for mass matrix
        /// @param[in] mesh the mesh
        /// @param[in] mass_matrix mass matrix on elements.
        /// @param[in] basis collocation points for Lagrange basis set.
        /// @param[in] quad quadrature rule for computing integrals.
        template <bool Diagonal>
        Projector(const Mesh2D& mesh, const MassMatrix<Diagonal>& mass_matrix, const QuadratureRule * basis, const QuadratureRule * quad=nullptr)
            : lf(mesh, basis, quad), m(&mass_matrix) {}

        /// @brief Computes the projection of f=f(x)
        /// @tparam Func invocable
        /// @param[in] f f(const double x[2], double F[n_var]) on exit F = f(x)
        /// @param[out] F The projection as a grid function. shape (n_var, n, n, n_elem) where n is the number of
        /// basis functions/collocation points.
        /// @param[in] n_var vector dimension of f
        template <typename Func>
        void operator()(Func f, double * F, int n_var = 1) const
        {
            lf(f, F, n_var);
            m->inv(F);
        }

        /// @brief computes the projection of f=f(x,u) where u is a grid
        /// function on the mesh. Assumes that f and u have the same vector
        /// dimension.
        /// @tparam Func invocable
        /// @param[in] f f(const double x[2], const double u[n_var], double F[n_var]) on
        /// exit F = f(x, u)
        /// @param[in] u Grid function. shape (n_var, n, n, n_elem) where n is the
        /// number of basis functions/collocation points.
        /// @param[out] F The projection as a grid function. shape (n_var, n, n, n_elem) where n is the number of
        /// basis functions/collocation points.
        /// @param[in] n_var vector dimension of f and u
        template <typename Func>
        void operator()(Func f, const double * u, double * F, int n_var = 1) const
        {
            lf(f, u, F, n_var);
            m->inv(F);
        }
    };
} // namespace dg


#endif