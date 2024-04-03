#ifndef DG_COVARIANT_NORMAL_PROLONGATOR_HPP
#define DG_COVARIANT_NORMAL_PROLONGATOR_HPP

#include "wdg_config.hpp"
#include "FEMVector.hpp"
#include "Mesh2D.hpp"
#include "Tensor.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    /// @brief computes the normal derivative at element faces in reference
    /// coordinates. This corresponds to one of the covariant derivatives
    /// (depending on the face). This quantity together with the face values
    /// (computed by FaceProlongator) which can be used to compute the
    /// tangential derivative (other covariant) can be used to reconstruct
    /// gradient at the face. The reason that this derivative is computed is
    /// because it can be represented exactly on the degrees of freedom of the
    /// face whereas the normal derivative in physical coordinates will be
    /// aliased since it is a higher order polynomial.
    class CovariantNormalProlongator
    {
    public:
        CovariantNormalProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type);

        ~CovariantNormalProlongator() = default;

        /// @brief computes the covariant normal derivative.
        /// @param n_var vector dimension of u
        /// @param u FEMVector
        /// @param uf FaceVector. On exit, uf has the covariant normal derivative.
        void action(int n_var, const double * u, double * uf) const;

        /// @brief computes the transpose of the operator. u <- u + E' * uf
        /// where E is `action`.
        /// @param n_var vector dimension of u
        /// @param uf FaceVector.
        /// @param u FEMVector. On exit, u <- u + E' * uf.
        void t(int n_var, const double * uf, double * u) const;
    
    protected:
        const int dim;
        const int n_elem;
        const int n_faces;
        const int n_basis;
        const FaceType face_type;

        ivec _v2e;
        dvec P;

    #ifdef WDG_USE_MPI
        const_ivec_wrapper lfp;
    #endif
    };
} // namespace dg


#endif