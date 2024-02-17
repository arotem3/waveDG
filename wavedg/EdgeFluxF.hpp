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
    class EdgeFluxF1D
    {
    public:
        EdgeFluxF1D(int n_var, const Mesh1D& mesh, FaceType face_type, const QuadratureRule * basis);

        /// @brief evaluates numerical flux on faces
        /// @tparam Func invocable as void(*)(double x, const double uL[n_var], const double uR[n_var], double fh[n_var])
        /// @param numerical_flux when invoked, numerical_flux set fh to the numerical flux given left (uL) and right (uR) values of the solution at the face at position x.
        /// @param uB Shape (n_var, 2, n_faces).
        /// @param fB Shape (n_var, 2, n_faces). It is safe to provide fB = uB to perform the action inplace on uB
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

} // namespace dg


#endif