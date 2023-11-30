#ifndef DG_FACE_PROJECTOR_HPP
#define DG_FACE_PROJECTOR_HPP

#include <vector>

#include "wdg_config.hpp"
#include "Tensor.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "MPI_Base.hpp"
#include "Operator.hpp"

namespace dg
{
    // projects grid functions to element faces
    class FaceProlongator : public Operator
    {
    protected:
        struct PersistantChannel
        {
            int partner; // rank to send/recv with
            ivec l2p; // local edge to send
            ivec p2l; // partner edges to recv
            mutable dvec send_buf;
            mutable dvec recv_buf;
        };

    public:
        /// @brief Prolongs element values to faces
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
        /// @param[in] n_var vector dimension of grid function
        // virtual void action(const double * u, double * uf) const = 0;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        virtual void t(const double * uf, double * u) const = 0;
    };

    class LobattoFaceProlongator : public FaceProlongator
    {
    private:
        const int n_elem;
        const int n_edges;
        const int n_colloc;
        const int n_var;
        Edge::EdgeType edge_type;
        icube v2e;

    #ifdef WDG_USE_MPI
        std::vector<int> local_face_pattern; // s+2*e if element is owned
        std::vector<PersistantChannel> channels;
        mutable RequestVec rreq;
        mutable RequestVec sreq;
    #endif

    public:
        /// @brief setup FaceProlongator
        /// @param mesh 
        /// @param basis 
        /// @param edge_type
        LobattoFaceProlongator(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type);

        /// @brief Prolongs element values to faces. With MPI, this function is blocking.
        /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
        /// @param[in] n_var vector dimension of grid function
        virtual void action(const double * u, double * uf) const override;

        /// @brief Applies the transpose of the prolongation operation.
        /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
        /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
        /// On exit, u = u + A' * uf where A is the prolongation operator.
        /// @param n_var vector dimension of grid function
        virtual void t(const double * uf, double * u) const override;
    };

    // class LegendreFaceProlongator : public FaceProlongator
    // {
    // private:
    //     const int n_elem;
    //     const int n_edges;
    //     const int n_colloc;
    //     Edge::EdgeType edge_type;
    //     Tensor<4, int> v2e;
    //     dvec P;
    
    // public:
    //     /// @brief setup FaceProlongator
    //     /// @param mesh 
    //     /// @param basis 
    //     /// @param edge_type
    //     LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type);

    //     /// @brief Prolongs element values to faces
    //     /// @param[in] u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
    //     /// @param[in,out] uf shape (2, n_var, n_colloc, n_edges). Edge values
    //     /// @param[in] n_var vector dimension of grid function
    //     virtual void action(const double * u, double * uf) const override;

    //     /// @brief Applies the transpose of the prolongation operation.
    //     /// @param uf shape (2, n_var, n_colloc, n_edges), Edge values
    //     /// @param u shape (n_var, n_colloc, n_colloc, n_elem). Element values.
    //     /// On exit, u = u + A' * uf where A is the prolongation operator.
    //     /// @param n_var vector dimension of grid function
    //     virtual void t(const double * uf, double * u) const override;
    // };

    inline std::unique_ptr<FaceProlongator> make_face_prolongator(int n_var, const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type)
    {
        if (basis->type == QuadratureRule::GaussLobatto)
        {
            return std::unique_ptr<FaceProlongator>(new LobattoFaceProlongator(n_var, mesh, basis, edge_type));
        }
        // else
        // {
        //     return std::unique_ptr<FaceProlongator>(new LegendreFaceProlongator(mesh, basis, edge_type));
        // }
    }
} // namespace dg

#endif