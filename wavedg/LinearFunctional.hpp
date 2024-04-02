#ifndef DG_BASIS_PRODUCT_HPP
#define DG_BASIS_PRODUCT_HPP

#include "wdg_config.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    /// @brief computes integrals of the form: \f$(f, \phi)\f$ for every basis function \f$\phi\f$ for every element.
    template <int Dim>
    class LinearFunctional
    {
    public:
        /// @brief constructs LinearFunctional
        /// @param[in] mesh
        /// @param[in] basis the basis set (defined as collocation on a quadrature rule).
        /// @param[in] quad_ quadrature rule for computing the integrals. If nullptr specified then quad_ = basis.
        template <typename Mesh>
        LinearFunctional(const Mesh& mesh, const QuadratureRule * basis, const QuadratureRule * quad_ =nullptr);

        /// @brief computes the inner product \f$(f, \phi)\f$ where \f$f=f(x)\f$ for every basis
        /// function \f$\phi\f$ on every element.
        /// @tparam Func invocable
        /// @param[in] f f(const double x[2], double F[n_var]) on exit F = f(x)
        /// @param[in] F shape (n_var, n, n, n_elem) where n is the number of
        /// basis functions/collocation points.
        /// @param[in] n_var vector dimension of f
        template <typename Func>
        void action(int n_var, Func f, double * F) const;

        /// @brief computes the inner product \f$(f(x, u), \phi)\f$ for every basis
        /// function phi on every element.
        /// @tparam Func invocable
        /// @param[in] f `f(const double x[2], const double u[n_var], double F[n_var])` on
        /// exit `F` \f$= f(x, u)\f$.
        /// @param[in] u shape `(n_var, n, n, n_elem)` where n is the number of basis
        /// functions/collocation points.
        /// @param[in] F shape `(n_var, n, n, n_elem)`.
        /// @param[in] n_var vector dimension of \f$f\f$ and \f$u\f$.
        template <typename Func>
        void action(int n_var, Func f, const double * u, double * F) const;

    private:
        const int n_elem;
        const int n_colloc;

        const QuadratureRule * quad;
        const int n_quad;
        
        const double * detJ_;
        const double * x_;

        dmat B;
    };

    typedef LinearFunctional<1> LinearFunctional1D;
    typedef LinearFunctional<2> LinearFunctional2D;

    template <> template <>
    LinearFunctional<2>::LinearFunctional(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad_)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          quad(quad_ ? quad_ : QuadratureRule::quadrature_rule(n_colloc)),
          n_quad(quad->n),
          B(n_quad, n_colloc)
    {
        auto& metrics = mesh.element_metrics(quad);
        detJ_ = metrics.measures();
        x_ = metrics.physical_coordinates();

        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);
    }

    template <> template <>
    LinearFunctional<1>::LinearFunctional(const Mesh1D& mesh, const QuadratureRule * basis, const QuadratureRule * quad_)
        : n_elem(mesh.n_elem()),
          n_colloc(basis->n),
          quad(quad_ ? quad_ : QuadratureRule::quadrature_rule(n_colloc)),
          n_quad(quad->n),
          B(n_quad, n_colloc)
    {
        auto& metrics = mesh.element_metrics(quad);
        detJ_ = metrics.jacobians();
        x_ = metrics.physical_coordinates();

        lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);
    }

    template <>
    template <typename Func>
    inline void LinearFunctional<1>::action(int n_var, Func f, double * F_) const
    {
        auto detJ = reshape(detJ_, n_quad, n_elem);
        auto xs = reshape(x_, n_quad, n_elem);

        auto F = reshape(F_, n_var, n_colloc, n_elem);

        dvec feval(n_var);
        dmat Fq(n_quad, n_var);

        auto W = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate and scale by detJ * w
            for (int i = 0; i < n_quad; ++i)
            {
                const double x[] = {xs(i, el)};
                f(x, feval);

                const double dx = detJ(i, el) * W(i);
                for (int d = 0; d < n_var; ++d)
                {
                    Fq(i, d) = feval(d) * dx;
                }
            }

            // integrate
            for (int k = 0; k < n_colloc; ++k)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double projF = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        projF += B(i, k) * Fq(i, d);
                    }
                    F(d, k, el) = projF;
                }
            }
        }
    }

    template <>
    template <typename Func>
    inline void LinearFunctional<2>::action(int n_var, Func f, double * F_) const
    {
         auto detJ = reshape(detJ_, n_quad, n_quad, n_elem);
        auto coo = reshape(x_, 2, n_quad, n_quad, n_elem);

        auto F = reshape(F_, n_var, n_colloc, n_colloc, n_elem);

        dcube Fq(n_var, n_quad, n_quad);
        dcube ProjF(n_var, n_quad, n_colloc);
        auto W = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate and scale by detJ
            for (int l = 0; l < n_quad; ++l)
            {
                for (int k = 0; k < n_quad; ++k)
                {
                    const double * x = &coo(0, k, l, el);
                    double * fq = &Fq(0, k, l);
                    f(x, fq);

                    const double w = detJ(k, l, el) * W(k) * W(l);
                    for (int d = 0; d < n_var; ++d)
                    {
                        Fq(d, k, l) *= w;
                    }
                }
            }

            // project along y
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int k = 0; k < n_quad; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double projF = 0.0;
                        for (int l = 0; l < n_quad; ++l)
                        {
                            projF += Fq(d, k, l) * B(l, j);
                        }
                        ProjF(d, k, j) = projF;
                    }
                }
            }

            // project along x
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double projF = 0.0;
                        for (int k = 0; k < n_quad; ++k)
                        {
                            projF += ProjF(d, k, j) * B(k, i);
                        }
                        F(d, i, j, el) = projF;
                    }
                }
            }
        }
    }

    template <>
    template <typename Func>
    inline void LinearFunctional<1>::action(int n_var, Func f, const double * u_, double * F_) const
    {
        auto detJ = reshape(detJ_, n_quad, n_elem);
        auto xs = reshape(x_, 2, n_quad, n_elem);

        auto F = reshape(F_, n_var, n_colloc, n_elem);
        auto u = reshape(u_, n_var, n_colloc, n_elem);

        dvec feval(n_var);
        dvec Uq(n_var);
        dmat Fq(n_quad, n_var);

        auto W = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                const double x[] = {xs(i, el)};
                
                // evaluate U
                for (int d = 0; d < n_var; ++d)
                {
                    double uq = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        uq += B(i, k) * u(d, k, el);
                    }
                    Uq(d) = uq;
                }

                // eval f
                f(x, Uq, feval);

                // scale by detJ * w
                for (int d = 0; d < n_var; ++d)
                {
                    Fq(i, d) = feval(d) * detJ(i, el) * W(i);
                }
            }

            // integrate
            for (int k = 0; k < n_colloc; ++k)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double projF = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        projF += Fq(i, d) * B(i, k);
                    }
                    F(d, k, el) = projF;
                }
            }
        }
    }

    template <>
    template <typename Func>
    inline void LinearFunctional<2>::action(int n_var, Func f, const double * u_, double * F_) const
    {
        auto detJ = reshape(detJ_, n_quad, n_quad, n_elem);
        auto coo = reshape(x_, 2, n_quad, n_quad, n_elem);

        auto F = reshape(F_, n_var, n_colloc, n_colloc, n_elem);
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);

        dcube Fq(n_var, n_quad, n_quad);
        dcube ProjF(n_var, n_quad, n_colloc);

        auto W = reshape(quad->w, n_quad);

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate f and scale by detJ
            for (int l = 0; l < n_quad; ++l)
            {
                for (int k = 0; k < n_quad; ++k)
                {
                    const double * x = &coo(0, k, l, el);
                    double * fq = &Fq(0, k, l);
                    const double * v = &u(0, k, l, el);
                    f(x, v, fq);

                    for (int d = 0; d < n_var; ++d)
                    {
                        Fq(d, k, l) *= detJ(k, l, el) * W(l) * W(k);
                    }
                }
            }

            // project along y
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int k = 0; k < n_quad; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double projF = 0.0;
                        for (int l = 0; l < n_quad; ++l)
                        {
                            projF += Fq(d, k, l) * B(l, j);
                        }
                        ProjF(d, k, j) = projF;
                    }
                }
            }

            // project along x
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double projF = 0.0;
                        for (int k = 0; k < n_quad; ++k)
                        {
                            projF += ProjF(d, k, j) * B(k, i);
                        }
                        ProjF(d, i, j, el) = projF;
                    }
                }
            }
        }
    }
} // namespace dg

#endif