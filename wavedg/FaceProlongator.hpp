#ifndef DG_FACE_PROJECTOR_HPP
#define DG_FACE_PROJECTOR_HPP

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"

namespace dg
{
    // projects grid functions to element faces
    class FaceProlongator
    {
    public:
        /// @brief Prolongs element values to faces
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
        /// @param[in] n_var vector dimension of grid function
        virtual void action(const double * u, double * uf, int n_var=1) const = 0;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        virtual void t(const double * uf, double * u, int n_var=1) const = 0;
    };

    class LobattoFaceProlongator : public FaceProlongator
    {
    private:
        const int n_elem;
        const int n_edges;
        const int n_colloc;
        Edge::EdgeType edge_type;
        Cube<int> v2e;

    public:
        /// @brief setup FaceProlongator
        /// @param mesh 
        /// @param basis 
        /// @param edge_type
        LobattoFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type);

        /// @brief Prolongs element values to faces
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
        /// @param[in] n_var vector dimension of grid function
        virtual void action(const double * u, double * uf, int n_var=1) const override;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        virtual void t(const double * uf, double * u, int n_var=1) const override;
    };

    class LegendreFaceProlongator : public FaceProlongator
    {
    private:
        const int n_elem;
        const int n_edges;
        const int n_colloc;
        Edge::EdgeType edge_type;
        Tensor<4, int> v2e;
        dvec P;
    
    public:
        /// @brief setup FaceProlongator
        /// @param mesh 
        /// @param basis 
        /// @param edge_type
        LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type);

        /// @brief Prolongs element values to faces
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
        /// @param[in] n_var vector dimension of grid function
        virtual void action(const double * u, double * uf, int n_var=1) const override;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        virtual void t(const double * uf, double * u, int n_var=1) const override;
    };

    inline std::unique_ptr<FaceProlongator> make_face_prolongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type)
    {
        if (basis->type == QuadratureRule::GaussLobatto)
        {
            return std::unique_ptr<FaceProlongator>(new LobattoFaceProlongator(mesh, basis, edge_type));
        }
        else
        {
            return std::unique_ptr<FaceProlongator>(new LegendreFaceProlongator(mesh, basis, edge_type));
        }
    }
} // namespace dg

#endif