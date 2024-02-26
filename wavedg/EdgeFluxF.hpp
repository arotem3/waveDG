#ifndef WDG_EDGE_FLUX_F_HPP
#define WDG_EDGE_FLUX_F_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "QuadratureRule.hpp"
#include "Mesh2D.hpp"
#include "Mesh1D.hpp"
#include "lagrange_interpolation.hpp"
#include "Operator.hpp"

namespace dg
{
    /// @brief Computes \f$f^\star v|_{\partial I}\f$.
    ///
    /// @details In the discretization of the nonlinear conservation law:
    /// $$u_t + f(u)_x = 0$$
    /// the DG weak form on each element \f$I\f$ is:
    /// $$\frac{d}{dt}(u, v)_I = (f, v_x)_I - f^\star v\bigg|_{\partial I}.$$
    /// Here \f$f^\star\f$ is the numerical flux.
    /// EdgeFlux1D computes the face terms:  \f$f^\star v|_{\partial I}\f$.
    class EdgeFluxF1D
    {
    public:
        /// @brief initializes EdgeFluxF1D
        /// @param n_var vector dimension of u
        /// @param mesh the 1D mesh
        /// @param face_type face type. INTERIOR or BOUNDARY
        /// @param basis collocation grid of basis functions
        EdgeFluxF1D(int n_var, const Mesh1D& mesh, FaceType face_type, const QuadratureRule * basis);

        /// @brief evaluates numerical flux on faces
        /// @tparam Func invocable as `void(*)(double x, const double uL[n_var], const double uR[n_var], double fh[n_var])`.
        /// @param numerical_flux when invoked, `numerical_flux` set `fh` to the
        /// numerical flux given left (`uL`) and right (`uR`) values of the solution
        /// at the face at position `x`.
        /// @param uB Shape `(n_var, 2, n_faces)`.
        /// @param fB Shape `(n_var, 2, n_faces)`. It is safe to provide `fB = uB` to perform the action inplace on `uB`.
        template <typename Func>
        void action(Func numerical_flux, const double * uB, double * fB) const;

    private:
        const FaceType face_type;
        const int n_var;
        const int n_faces;
        
        dvec x;
        mutable dvec fh;
        mutable dvec uL;
        mutable dvec uR;
    };

    /// @brief Computes \f$\langle (\vec{n}\cdot \vec{F})^\star, v \rangle\f$.
    ///
    /// @details In the discretization of the nonlinear conservation law:
    /// $$u_t + \nabla \cdot \vec{F}(u) = 0$$
    /// where \f$\vec{F}(u) = [f(u), g(u)]^T\f$,
    /// the DG weak form on each element is:
    /// $$\frac{d}{dt}(u, v) = (\vec{F}, \nabla v) - \langle (\vec{n}\cdot \vec{F})^\star, v \rangle.$$
    /// Here \f$(\vec{n}\cdot \vec{F})^\star\f$ is the numerical flux.
    /// EdgeFluxF2D computes the face integral \f$\langle (\vec{n}\cdot \vec{F})^\star, v \rangle\f$.
    /// The integral can be computed using the quadrature rule of the basis
    /// function, or by supplying a higher order quadrature rule. If using the
    /// quadrature rule of the basis, no projections are used and thus there may
    /// be aliasing errors when \f$\vec{F}\f$ is nonlinear or has variable
    /// coefficients, but the computation is faster.
    /// @tparam ApproxQuadrature 
    template <bool ApproxQuadrature>
    class EdgeFluxF2D
    {
    public:
        /// @brief initializes EdgeFlux2D
        /// @param n_var vector dimension of u
        /// @param mesh the 2D mesh
        /// @param face_type face type. Either INTERIOR or BOUNDARY.
        /// @param basis collocation grid of basis functions.
        /// @param quad quadrature rule for computing integrals.
        EdgeFluxF2D(int n_var, const Mesh2D& mesh, FaceType face_type, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        /// @brief Computes 
        /// @tparam Func invocable as `void(*)(const double x[2], const double n[2], const double u_int[n_var], const double u_ext[n_var], double f_star[n_var])`
        /// @param numerical_flux `numerical_flux(x, n, u_int, u_ext, f_star)` on
        /// exit: `f_star[i]` \f$=(\vec{n}\cdot\vec{F})^\star_i\f$ computed using the element interior
        /// (`u_int`) and elemenet exterior (`u_ext`) value.
        /// @param uB the face values of u. Shape `(n_colloc, n_var, 2, n_edges)`.
        /// @param fB the numerical flux on the faces.
        /// Shape `(n_colloc, n_var, 2, n_edges)`.
        /// It is safe to set `fB = uB` to perform the computation in place on `uB`.
        template <typename Func>
        void action(Func numerical_flux, const double * uB, double * fB) const;

    private:
        const FaceType face_type;
        const int n_var;
        const int n_colloc;
        const int n_faces;

        const QuadratureRule * const quad;
        const int n_quad;

        const_dcube_wrapper X;
        const_dcube_wrapper normals;
        const_dmat_wrapper meas;

        dmat P;
        dmat Pt;

        mutable dmat F;
        mutable dvec fh;
        mutable dvec uf[2];
    };

    template <typename Func>
    void EdgeFluxF1D::action(Func flux, const double * uB_, double * fB_) const
    {
        auto uB = reshape(uB_, n_var, 2, n_faces);
        auto fB = reshape(fB_, n_var, 2, n_faces);

        for (int e = 0; e < n_faces; ++e)
        {
            // copy to face values to uf
            for (int d = 0; d < n_var; ++d)
            {
                uL(d) = uB(d, 0, e);
                uR(d) = uB(d, 1, e);
            }

            // evaluate numerical flux
            const double xi = x(e);
            flux(xi, const_cast<const double *>(uL.data()), const_cast<const double *>(uR.data()), fh.data());

            // copy fh to fB
            for (int d = 0; d < n_var; ++d)
            {
                fB(d, 0, e) =  fh(d);
                fB(d, 1, e) = -fh(d);
            }
        }
    }

    template <> template <typename Func>
    void EdgeFluxF2D<true>::action(Func flux, const double * uB_, double * fB_) const
    {
        auto uB = reshape(uB_, n_colloc, n_var, 2, n_faces);
        auto fB = reshape(fB_, n_colloc, n_var, 2, n_faces);

        double x[2];
        double n[2];

        auto W = reshape(quad->w, n_quad);

        for (int e = 0; e < n_faces; ++e)
        {
            for (int i = 0; i < n_colloc; ++i)
            {
                x[0] = X(0, i, e);
                x[1] = X(1, i, e);

                n[0] = normals(0, i, e);
                n[1] = normals(1, i, e);

                for (int d = 0; d < n_var; ++d)
                {
                    uf[0](d) = uB(i, d, 0, e);
                    uf[1](d) = uB(i, d, 1, e);
                }

                flux(const_cast<const double * const>(x),
                     const_cast<const double * const>(n),
                     const_cast<const double * const>(uf[0].data()),
                     const_cast<const double * const>(uf[1].data()),
                     const_cast<double * const>(fh.data()));
                
                for (int d = 0; d < n_var; ++d)
                {
                    double fl = fh(d) * W(i) * meas(i, e);
                    fB(i, d, 0, e) =  fl;
                    fB(i, d, 1, e) = -fl;
                }
            }
        }
    }

    template <> template <typename Func>
    void EdgeFluxF2D<false>::action(Func flux, const double * uB_, double * fB_) const
    {
        auto uB = reshape(uB_, n_colloc, n_var, 2, n_faces);
        auto fB = reshape(fB_, n_colloc, n_var, 2, n_faces);
        
        double x[2];
        double n[2];

        auto W = reshape(quad->w, n_quad);

        for (int e = 0; e < n_faces; ++e)
        {
            for (int i = 0; i < n_quad; ++i)
            {
                x[0] = X(0, i, e);
                x[1] = X(1, i, e);

                n[0] = normals(0, i, e);
                n[1] = normals(1, i, e);

                // evaluate u on quadrature point
                for (int s = 0; s < 2; ++s)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double ud = 0;
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            ud += Pt(k, i) * uB(k, d, s, e);
                        }
                        uf[s](d) = ud;
                    }
                }

                // evaluate numerical flux
                flux(const_cast<const double * const>(x),
                     const_cast<const double * const>(n),
                     const_cast<const double * const>(uf[0].data()),
                     const_cast<const double * const>(uf[1].data()),
                     const_cast<double * const>(fh.data()));
                
                for (int d = 0; d < n_var; ++d)
                {
                    F(i, d) = fh(d);
                }
            }

            // integral (f, v)
            for (int d = 0; d < n_var; ++d)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    double fl = 0.0;
                    for (int i = 0; i < n_quad; ++i)
                    {
                        const double ds = W(i) * meas(i, e);
                        fl += F(i, d) * P(i, k) * ds;
                    }

                    fB(k, d, 0, e) =  fl;
                    fB(k, d, 1, e) = -fl;
                }
            }
        }
    }
} // namespace dg


#endif