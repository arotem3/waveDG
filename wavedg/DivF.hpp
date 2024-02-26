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
    /// @brief Computes \f$(f, v_x)\f$ where \f$f = f(x, u)\f$.
    ///
    /// @details In the discretization of the nonlinear conservation law:
    /// $$u_t + f(u)_x = 0$$
    /// the DG weak form on each element \f$I\f$ is:
    /// $$\frac{d}{dt}(u, v)_I = (f, v_x)_I - f^\star v\bigg|_{\partial I}.$$
    /// Here \f$f^\star\f$ is the numerical flux.
    /// The DivF2D class implements the volume term: \f$(f, v_x)\f$.
    /// The integral can be computed using the quadrature rule of the basis
    /// function, or by supplying a higher order quadrature rule. If using the
    /// quadrature rule of the basis, no projections are used and thus there may
    /// be aliasing errors when \f$f\f$ is nonlinear or has variable
    /// coefficients, but the computation is faster.
    /// @tparam ApproxQuadrature    
    template <bool ApproxQuadrature>
    class DivF1D
    {
    public:
        /// @brief initializes DivF1D
        /// @param nvar vector dimension of u
        /// @param mesh the 1D mesh
        /// @param basis collocation grid of basis functions
        /// @param quad quadrature rule for computing integrals
        DivF1D(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        /// @brief computes \f$(f, v_x)\f$.
        /// @tparam Func invocable as `F(const double x, const double u[n_var], double f[n_var])`
        /// @param F `F(x, u, f)` overwrites `f` so that `f[d]` \f$f(x, u)_d\f$ for `d=0,...,n_var-1`.
        /// @param u solution vector. Shape `(n_var, n_colloc, n_elem)`.
        /// @param divF \f$(f, v_x)\f$ for every basis function v. Shape `(n_var, n_colloc, n_elem)`.
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

    /// @brief Computes \f$(\vec{F}, \nabla v)\f$ where \f$\vec{F} = \vec{F}(x, u)\f$.
    ///
    /// @details In the discretization of the nonlinear conservation law:
    /// $$u_t + \nabla \cdot \vec{F}(u) = 0$$
    /// where \f$\vec{F}(u) = [f(u), g(u)]^T\f$,
    /// the DG weak form on each element is:
    /// $$\frac{d}{dt}(u, v) = (\vec{F}, \nabla v) - \langle (\vec{n}\cdot \vec{F})^\star, v \rangle.$$
    /// Here \f$(\vec{n}\cdot \vec{F})^\star\f$ is the numerical flux.
    /// The DivF2D class implements the volume term: \f$(\vec{F}, \nabla v)\f$.
    /// The integral can be computed using the quadrature rule of the basis
    /// function, or by supplying a higher order quadrature rule. If using the
    /// quadrature rule of the basis, no projections are used and thus there may
    /// be aliasing errors when \f$\vec{F}\f$ is nonlinear or has variable
    /// coefficients, but the computation is faster.
    /// @tparam ApproxQuadrature 
    template <bool ApproxQuadrature>
    class DivF2D
    {
    public:
        /// @brief initializes DivF2D
        /// @param n_var vector dimension of u
        /// @param mesh the 2D mesh
        /// @param basis collocation grid of basis functions
        /// @param quad quadrature rule for computing integrals. If ApproxQuad == false and quad == nullptr,
        /// then quad = QuadratureRule::quadrature_rule(basis->n);
        DivF2D(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        /// @brief compute \f$(\vec{F}, \nabla v)\f$.
        /// @tparam Func invocable as `void(*)(const double x[2], const double u[n_var], const double f[2*n_var])`
        /// @param[in] F `F(x, u, f)` overwrites `f[i]` \f$= f(x, u)_i\f$ and `f[i+n_var]` \f$= g(x, u)_i\f$ for `i=0,...,n_var-1`.
        /// @param[in,out] u DG grid function. Shape `(n_var, n_colloc, n_colloc, n_elem)`.
        /// @param[in,out] divF On exit, `divF` \f$= (\vec{F}, \nabla v)\f$. Shape `(n_var, n_colloc, n_colloc, n_elem)`.
        template <typename Func>
        void action(Func F, const double * u, double * divF) const;

    private:
        const int n_var;
        const int n_colloc;
        const int n_elem;

        const QuadratureRule * const quad;
        const int n_quad;

        TensorWrapper<5, const double> J;
        TensorWrapper<4, const double> X;
        
        dmat D;
        dmat Dt;
        dmat P;
        dmat Pt;

        mutable dvec Uq;
        mutable dmat Fq;
        mutable Tensor<4, double> F;

        mutable dvec work1;
        mutable dvec work2;
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

    template <> template <typename Func>
    void DivF2D<true>::action(Func f, const double * u_, double * divF_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto divF = reshape(divF_, n_var, n_colloc, n_colloc, n_elem);

        auto w = reshape(quad->w, n_colloc);
        
        double x[2];

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate flux
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    // metric map
                    const double w_ij = w(i) * w(j);
                    const double Y_eta = J(1, 1, i, j, el) * w_ij;
                    const double X_eta = J(0, 1, i, j, el) * w_ij;
                    const double Y_xi  = J(1, 0, i, j, el) * w_ij;
                    const double X_xi  = J(0, 0, i, j, el) * w_ij;

                    // coordinates of collocation node
                    x[0] = X(0, i, j, el);
                    x[1] = X(1, i, j, el);

                    // u|node
                    for (int d = 0; d < n_var; ++d)
                        Uq(d) = u(d, i, j, el);
                    
                    // evaluate physical flux
                    f(const_cast<const double* const>(x),
                      const_cast<const double* const>(Uq.data()),
                      const_cast<double* const>(Fq.data()));

                    // compute contravariant flux
                    for (int d = 0; d < n_var; ++d)
                    {
                        F(0, i, d, j) =  Y_eta * Fq(d, 0) - X_eta * Fq(d, 1); // <- indexing is different for F(0) and F(1) for better striding
                        F(1, j, d, i) = -Y_xi  * Fq(d, 0) + X_xi  * Fq(d, 1);
                    }
                }
            }

            // integral (F, grad v)
            for (int j = 0; j < n_colloc; ++j)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double dF = 0.0;
                        for (int l = 0; l < n_colloc; ++l)
                        {
                            dF += D(l, i) * F(0, l, d, j) + D(l, j) * F(1, l, d, i);
                        }
                        divF(d, i, j, el) += dF;
                    }
                }
            }
        }
    }

    template <> template <typename Func>
    void DivF2D<false>::action(Func f, const double * u_, double * divF_) const
    {
        auto u = reshape(u_, n_var, n_colloc, n_colloc, n_elem);
        auto divF = reshape(divF_, n_var, n_colloc, n_colloc, n_elem);

        const int n_quad = quad->n;

        auto w = reshape(quad->w, n_quad);
        
        double x[2];
        auto Pu = reshape(work1, n_colloc, n_var, n_quad);
        auto Df = reshape(work1, n_quad, n_var, n_colloc);
        auto Pg = reshape(work2, n_quad, n_var, n_colloc);

        for (int el = 0; el < n_elem; ++el)
        {
            // evaluate F(u) on quadrature points
            for (int i = 0; i < n_quad; ++i)
            {
                for (int l = 0; l < n_colloc; ++l)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double pu = 0.0;
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            pu += Pt(k, i) * u(d, k, l, el);
                        }
                        Pu(l, d, i) = pu;
                    }
                }
            }

            for (int j = 0; j < n_quad; ++j)
            {
                for (int i = 0; i < n_quad; ++i)
                {
                    // metric map
                    const double w_ij = w(i) * w(j);
                    const double Y_eta = J(1, 1, i, j, el) * w_ij;
                    const double X_eta = J(0, 1, i, j, el) * w_ij;
                    const double Y_xi  = J(1, 0, i, j, el) * w_ij;
                    const double X_xi  = J(0, 0, i, j, el) * w_ij;

                    // coordinates of collocation node
                    x[0] = X(0, i, j, el);
                    x[1] = X(1, i, j, el);

                    // evaluate u on quadrature points
                    for (int d = 0; d < n_var; ++d)
                    {
                        double uq = 0.0;
                        for (int l = 0; l < n_colloc; ++l)
                        {
                            uq += Pt(l, j) * Pu(l, d, i);
                        }
                        Uq(d) = uq;
                    }

                    // evaluate physical flux
                    f(const_cast<const double* const>(x),
                      const_cast<const double* const>(Uq.data()),
                      const_cast<double* const>(Fq.data()));

                    // compute contravariant flux
                    for (int d = 0; d < n_var; ++d)
                    {
                        F(0, i, d, j) =  Y_eta * Fq(d, 0) - X_eta * Fq(d, 1);
                        F(1, i, d, j) = -Y_xi  * Fq(d, 0) + X_xi  * Fq(d, 1);
                    }
                }
            }

            // integral (F, grad v)
            for (int j = 0; j < n_quad; ++j)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double df = 0.0;
                        double pg = 0.0;
                        for (int i = 0; i < n_quad; ++i)
                        {
                            df += D(i, k) * F(0, i, d, j);
                            pg += P(i, k) * F(1, i, d, j);
                        }
                        Df(j, d, k) = df;
                        Pg(j, d, k) = pg;
                    }
                }
            }

            for (int l = 0; l < n_colloc; ++l)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double dF = 0.0;
                        for (int j = 0; j < n_quad; ++j)
                        {
                            dF += P(j, l) * Df(j, d, k) + D(j, l) * Pg(j, d, k);
                        }
                        divF(d, k, l, el) += dF;
                    }
                }
            }
        }
    }
} // namespace dg

#endif