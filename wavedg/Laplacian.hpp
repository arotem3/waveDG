#ifndef DG_LAPLACIAN_HPP
#define DG_LAPLACIAN_HPP

#include "wdg_config.hpp"
#include "Operator.hpp"
#include "Mesh2D.hpp"
#include "QuadratureRule.hpp"
#include "Tensor.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    /// @brief computes the bilinear form: (grad u, grad v).
    template <bool ApproxQuadrature>
    class Laplacian : public Operator
    {
    public:
        Laplacian(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad = nullptr);

        ~Laplacian() = default;

        /// @brief computes du <- du + (grad u, grad v)
        /// @param u 
        /// @param du 
        void action(const double * u, double * du) const override
        {
            action(1, u, du);
        }

        /// @brief computes du <- du + (grad u, grad v)
        /// @param n_var vector dimension of u. The Laplacian is applied to each dimension.
        /// @param u 
        /// @param du 
        void action(int n_var, const double * u, double * du) const override;

    private:
        const int dim;
        const int n_basis;
        const int n_elem;

        const int n_quad;

        dmat D; // (n_quad, n_colloc)
        dmat Dt; // transpose of D
        dmat P; //(n_quad, n_colloc)
        dmat Pt; // transpose of P
        dvec _op;
    };
} // namespace dg

#endif
