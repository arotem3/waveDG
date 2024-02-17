#ifndef WDG_DIV_F_HPP
#define WDG_DIV_F_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "QuadratureRule.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
#include "lagrange_interpolation.hpp"
#include "Operator.hpp"

namespace dg
{
    template <bool ApproxQuadrature>
    class DivF1D
    {
    public:
        DivF1D(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        /// @brief computes (F(u), grad(v)) = (f, v_x).
        /// @tparam Func invocable as F(const double x, const double u[n_var], double f[n_var])
        /// @param F F(x, u, f) overwrites f so that f[d] = f(x, u)[d].
        /// @param u solution vector. Shape (n_var, n_colloc, n_elem)
        /// @param divF (f, v_x) for every basis function v. Shape (n_var, n_colloc, n_elem)
        template <typename Func>
        void action(Func F, const double * u, double * divF) const;

    private:
        const int n_var;
        const int n_colloc;
        const int n_elem;

        const QuadratureRule * quad;
        const_dmat_wrapper x;

        dmat D; // (n_quad, n_colloc)
        dmat Pt; // transpose of P

        mutable dvec Uq;
        mutable dvec Fq;
        mutable dmat F;
    };

    template <> template <typename Func>
    void DivF1D<true>::action(Func f, const double * u_, double * divF_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_elem);
        auto divF = reshape(divF_, n_var, n_colloc, n_elem);

        auto w = reshape(quad->w, n_colloc);

        for (int el = 0; el < n_elem; ++el)
        {
            // compute flux
            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                    Uq(d) = u(d, i, el);
                const double xi = x(i, el);

                f(xi, const_cast<const double*>(Uq.data()), Fq.data());

                for (int d = 0; d < n_var; ++d)
                    F(i, d) = Fq(d);
            }

            // integrate
            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double r = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        r += D(k, i) * F(k, d) * w(k);
                    }
                    divF(d, i, el) += r;
                }
            }
        }
    }

    template <> template <typename Func>
    void DivF1D<false>::action(Func f, const double * u_, double * divF_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_elem);
        auto divF = reshape(divF_, n_var, n_colloc, n_elem);

        const int n_quad = quad->n;
        auto w = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                // evaluate u on quadrature point
                for (int d = 0; d < n_var; ++d)
                {
                    double uq = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        uq += Pt(k, i) * u(d, k, el);
                    }
                    Uq(d) = uq;
                }

                // evaluate flux
                const double xi = x(i, el);
                f(xi, const_cast<const double*>(Uq.data()), Fq.data());

                for (int d = 0; d < n_var; ++d)
                    F(i, d) = Fq(d);
            }

            // integrate (F, v_x)
            for (int k = 0; k < n_colloc; ++k)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double r = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        r += D(i, k) * F(i, d) * w(i);
                    }
                    divF(d, k, el) += r;
                }
            }
        }
    }
} // namespace dg

#endif