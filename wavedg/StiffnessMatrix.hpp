#ifndef WDG_STIFFNESS_MATRIX_HPP
#define WDG_STIFFNESS_MATRIX_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Operator.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    template <bool ApproxQuadrature>
    class StiffnessMatrix : public Operator
    {
    public:
        StiffnessMatrix(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad = nullptr);

        ~StiffnessMatrix() = default;

        void action(int n_var, const double * u, double * Au) const override;

        void action(const double * u, double * Au) const override;

    private:
        const int n_elem;
        const int n_basis;
        const int n_quad;

        dmat P;
        dmat D;

        Tensor<4, double> G;
    };
} // namespace dg

#endif
