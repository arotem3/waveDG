#ifndef DG_FACE_PROJECTOR_HPP
#define DG_FACE_PROJECTOR_HPP

#include <vector>
#include <map>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh1D.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "MPI_Base.hpp"
#include "Operator.hpp"
#include "FEMVector.hpp"

namespace dg
{
    // projects grid functions to element faces
    class FaceProlongator
    {
    protected:
        const int dim;
        const int n_elem;
        const int n_edges;
        const int n_colloc;
        const FaceType face_type;

        ivec _v2e;

    #ifdef WDG_USE_MPI
        const_ivec_wrapper lfp;
    #endif

    public:
        FaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type);
        FaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType face_type);

        virtual ~FaceProlongator() = default;

        /// @brief Prolongs element values to faces.
        /// @param[in] n_var vector dimension of u.
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (n_colloc, n_var, 2, n_edges). Edge values.
        virtual void action(int n_var, const double * u, double * uf) const = 0;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param[in] n_var vector dimension of u
        /// @param[in] uf shape (n_colloc, n_var, 2, n_elem). Edge values.
        /// @param[in,out] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        virtual void t(int n_var, const double * uf, double * u) const = 0;
    };

    class LobattoFaceProlongator : public FaceProlongator
    {
    public:
        LobattoFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type);

        LobattoFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType face_type);

        ~LobattoFaceProlongator() = default;

        /// @brief Prolongs element values to faces.
        /// @param[in] n_var vector dimension of u.
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (n_colloc, n_var, 2, n_edges). Edge values.
        virtual void action(int n_var, const double * u, double * uf) const override;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param[in] n_var vector dimension of u
        /// @param[in] uf shape (n_colloc, n_var, 2, n_elem). Edge values.
        /// @param[in,out] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        virtual void t(int n_var, const double * uf, double * u) const override;
    };

    class LegendreFaceProlongator : public FaceProlongator
    {
    private:
        dvec P;
    
    public:
        LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type);

        LegendreFaceProlongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType face_type);

        /// @brief Prolongs element values to faces.
        /// @param[in] n_var vector dimension of u.
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (n_colloc, n_var, 2, n_edges). Edge values.
        virtual void action(int n_var, const double * u, double * uf) const override;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param[in] n_var vector dimension of u
        /// @param[in] uf shape (n_colloc, n_var, 2, n_elem). Edge values.
        /// @param[in,out] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        virtual void t(int n_var, const double * uf, double * u) const override;
    };

    inline std::unique_ptr<FaceProlongator> make_face_prolongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type)
    {
        if (basis->type == QuadratureRule::GaussLobatto)
            return std::unique_ptr<FaceProlongator>(new LobattoFaceProlongator(mesh, basis, face_type));
        else
            return std::unique_ptr<FaceProlongator>(new LegendreFaceProlongator(mesh, basis, face_type));
    }

    inline std::unique_ptr<FaceProlongator> make_face_prolongator(const Mesh1D& mesh, const QuadratureRule * basis, FaceType face_type)
    {
        if (basis->type == QuadratureRule::GaussLobatto)
            return std::unique_ptr<FaceProlongator>(new LobattoFaceProlongator(mesh, basis, face_type));
        else
            return std::unique_ptr<FaceProlongator>(new LegendreFaceProlongator(mesh, basis, face_type));
    }
} // namespace dg

#endif