#ifndef DG_BASIS_PRODUCT_HPP
#define DG_BASIS_PRODUCT_HPP

#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    // computes integrals of the form: (f, phi) for every basis function phi for
    // every element.
    class BasisProduct
    {
    private:
        const int n_elem;
        const int n_colloc;
        const int n_quad;
        const double * detJ_;
        const double * x_;
        const QuadratureRule * quad;
        dmat B;
    
    public:
        BasisProduct(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad_)
            : n_elem(mesh.n_elem()),
              n_colloc(basis->n),
              n_quad(quad_->n),
              quad(quad_),
              B(n_quad, n_colloc)
        {
            detJ_ = mesh.element_measures(quad);
            x_ = mesh.element_physical_coordinates(quad);

            lagrange_basis(B.data(), n_colloc, basis->x, n_quad, quad->x);
        }

        /// @brief computes the inner product (f(x), phi) for every basis
        /// function phi on every element.
        /// @tparam Func invocable
        /// @param f f(const double x[2], double F[n_var]) on exit F = f(x)
        /// @param F shape (n_var, n, n, n_elem) where n is the number of
        /// basis functions/collocation points.
        /// @param n_var vector dimension of f
        template <typename Func>
        void operator()(Func f, double * F_, int n_var = 1) const
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

        /// @brief computes the inner product (f(x, u), phi) for every basis
        /// function phi on every element.
        /// @tparam Func invocable
        /// @param f f(const double x[2], const double u[n_var], double F[n_var]) on
        /// exit F = f(x, u)
        /// @param u shape (n_var, n, n, n_elem) where n is the number of basis
        /// functions/collocation points.
        /// @param F shape (n_var, n, n, n_elem).
        /// @param n_var vector dimension of f
        template <typename Func>
        void operator()(Func f, const double * u_, double * F_, int n_var=1) const
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
    };
} // namespace dg

#endif