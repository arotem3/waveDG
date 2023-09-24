#ifndef DG_PROJECT_HPP
#define DG_PROJECT_HPP

#include "config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "MassMatrix.hpp"
#include "BasisProduct.hpp"

namespace dg
{
    // projects functions onto the basis set on every element.
    template <bool Diagonal>
    class Projector
    {
    private:
        BasisProduct bprod;
        const MassMatrix<Diagonal>& m;

    public:
        Projector(const Mesh2D& mesh, const MassMatrix<Diagonal>& mass_matrix, const QuadratureRule * basis, const QuadratureRule * quad)
            : bprod(mesh, basis, quad), m(mass_matrix) {}

        template <typename Func>
        void operator()(Func f, double * F, int n_var = 1) const
        {
            bprod(f, F, n_var);
            m.inv(F);
        }

        template <typename Func>
        void operator()(Func f, const double * u, double * F, int n_var = 1) const
        {
            bprod(f, u, F, n_var);
            m.inv(F);
        }
    };
} // namespace dg


#endif