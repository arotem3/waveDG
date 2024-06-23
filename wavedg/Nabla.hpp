#ifndef WDG_NABLA_HPP
#define WDG_NABLA_HPP

#include "wdg_config.hpp"
#include "Operator.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "QuadratureRule.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    template <bool ApproxQuad>
    class Nabla
    {
    public:
        Nabla(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad = nullptr);

        /// @brief elementwise (grad u, phi) for scalar field u
        /// @param u FEMVector with n_var = 1.
        /// @param grad_u FEMVector with n_var = 2. On exit, the gradient of u.
        void grad(const double * u, double * grad_u) const;

        /// @brief elementwise (curl u, phi) where u is vector field in the
        /// xy-plane so that curl(u) is aligned with the z-axis.
        /// @param u FEMVector with n_var = 2. Vector field.
        /// @param curl_u FEMVector with n_var = 1. On exit, the z-component of the curl of u.
        void xycurl(const double * u, double * curl_u) const;

        /// @brief elementwise (curl u, phi) where u is a vector field aligned
        /// with the z-axis so that curl(u) is in the xy-plane.
        /// @param u FEMVector with n_var = 1. z-component of vector field.
        /// @param curl_u FEMVector with n_var = 2. On exit, the curl of u in xy-plane.
        void zcurl(const double * u, double * curl_u) const;

        /// @brief elementwise (div u, phi) for vector field u
        /// @param u FEMVector with n_var = 2.
        /// @param div_u FEMVector with n_var = 1. On exit, the divergence of u.
        void div(const double * u, double * div_u) const;
    
    public:
        const int n_elem;
        const int n_basis;

        dmat D;
        TensorWrapper<5, const double> J;
        const_dvec_wrapper w;
    };
} // namespace dg

#endif
