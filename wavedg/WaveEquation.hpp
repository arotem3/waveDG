#ifndef DG_WAVE_EQUATION_HPP
#define DG_WAVE_EQUATION_HPP

#include "wdg_config.hpp"
#include "Operator.hpp"
#include "Mesh2D.hpp"
#include "FaceProlongator.hpp"
#include "Div.hpp"
#include "EdgeFlux.hpp"

namespace dg
{
    /// @brief DG discretization of the wave equation:
    /// $$p_t + \nabla\cdot\vec{u} = 0,$$
    /// $$\vec{u}_t + \nabla p = 0.$$
    ///
    /// @details See Advection for details. This class specializes Advection to the wave equation.
    /// @tparam ApproxQuadrature 
    template <bool ApproxQuadrature>
    class WaveEquation : public Operator
    {
    public:
        /// @brief initialize DG discretization of wave equation
        /// @param mesh the 2d mesh
        /// @param basis basis function
        /// @param quad quadrature rule
        WaveEquation(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        ~WaveEquation() = default;

        /// @brief Dw <- [(u, grad phi) - <n.u*, phi>, (p, grad phi) - <n p*, phi>]
        /// @param[in] w wave equation discretization: w = [p, u]. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Dw Shape (3, n_colloc, n_colloc, n_elem)
        void action(const double * w, double * Dw) const override;

    private:
        std::unique_ptr<Operator> div;
        std::unique_ptr<Operator> flx;
        std::unique_ptr<FaceProlongator> prol;

        mutable dvec uI;
    };

    template <bool ApproxQuadrature>
    WaveEquation<ApproxQuadrature>::WaveEquation(const Mesh2D& mesh, const QuadratureRule * basis, const QuadratureRule * quad)
    {
        const int ne = mesh.n_edges(Edge::INTERIOR);
        const int n_colloc = basis->n;

        prol = make_face_prolongator(3, mesh, basis, Edge::INTERIOR);

        const double a[] = {
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,

            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0
        }; // wave equation as conservation law: p_t + div(u) == 0, u_t + grad(u) == 0.

        div.reset(new Div<ApproxQuadrature>(3, mesh, basis, a, true, quad));

        // flx.reset(new EdgeFlux<ApproxQuadrature>(3, mesh, Edge::INTERIOR, basis, a, true, -1.0, -0.5, quad));
        flx.reset(new EdgeFlux<ApproxQuadrature>(3, mesh, Edge::INTERIOR, basis, a, true, -1.0, 0.0, quad));

        uI.reshape(2 * 3 * n_colloc * ne);
    }

    template <bool ApproxQuadrature>
    void WaveEquation<ApproxQuadrature>::action(const double * u, double * divF) const
    {
        div->action(u, divF);

        prol->action(u, uI);
        flx->action(uI, uI);
        prol->t(uI, divF);
    }

    inline void reflect(double * v_ext, const double * v_int, const double * n)
    {
        v_ext[0] = v_int[0];
        v_ext[1] = (n[1]*n[1] - n[0]*n[0]) * v_int[1] - 2.0*n[0]*n[1] * v_int[2];
        v_ext[2] = -2.0*n[0]*n[1] * v_int[1] + (n[0]*n[0] - n[1]*n[1]) * v_int[2];
    }

    inline void absorb(double * v_ext, const double * v_int, const double * n)
    {
        double outgoing = v_int[0] + n[0] * v_int[1] + n[1] * v_int[2];
        double tangential_velocity = n[1] * v_int[1] - n[0] * v_int[2];
        v_ext[1] =  n[1] * tangential_velocity + 0.5 * n[0] * outgoing;
        v_ext[2] = -n[0] * tangential_velocity + 0.5 * n[1] * outgoing;
        v_ext[0] = n[0] * v_ext[1] + n[1] * v_ext[2];
    }

    template <bool ApproxQuadrature>
    class WaveBC : public Operator
    {
    public:
        /// @brief wave equation boundary conditions
        /// @param mesh 2d mesh
        /// @param bc Specifies the boundary condition on each boundary edge as either absorbing (0) or reflecting/Neumann (1).
        /// @param basis collocation points of Lagrange basis functions
        /// @param quad quadrature rule
        WaveBC(const Mesh2D& mesh, const int * bc, const QuadratureRule * basis, const QuadratureRule * quad=nullptr);

        ~WaveBC() = default;

        /// @brief applies boundary conditions Bw <- B(w)
        /// @param[in] w solution vector. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Bw output. Shape (3, n_colloc, n_colloc, n_elem)
        void action(const double * w, double * Bw) const override;

    private:
        const int nB;
        const int n_colloc;
        
        std::unique_ptr<Operator> flx;
        std::unique_ptr<FaceProlongator> prol;
        
        const_dcube_wrapper n;
        ivec bc;
        
        mutable Tensor<4,double> uB;
    };

    template <bool ApproxQuadrature>
    WaveBC<ApproxQuadrature>::WaveBC(const Mesh2D& mesh, const int * bc_, const QuadratureRule * basis, const QuadratureRule * quad)
        : nB(mesh.n_edges(Edge::BOUNDARY)),
          n_colloc(basis->n)
    {
        bc.reshape(nB);
        for (int i=0; i < nB; ++i)
        {
        #ifdef WDG_DEBUG
            if ((bc_[i] != 0) && (bc_[i] != 1))
                wdg_error("WaveBC error: specified boundary condition not implemented.");
        #endif
            bc(i) = bc_[i];
        }

        const double * n_ = mesh.edge_metrics(basis, Edge::BOUNDARY).normals();
        n = reshape(n_, 2, n_colloc, nB);

        prol = make_face_prolongator(3, mesh, basis, Edge::BOUNDARY);

        const double a[] = {
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 0.0, 0.0,

            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0
        };

        flx.reset(new EdgeFlux<ApproxQuadrature>(3, mesh, Edge::BOUNDARY, basis, a, true, -1.0, -0.5, quad));

        uB.reshape(n_colloc, 3, 2, nB);
    }

    template <bool ApproxQuadrature>
    void WaveBC<ApproxQuadrature>::action(const double * u, double * divF) const
    {
        // prolongate face values
        prol->action(u, uB);

        // compute exterior values
        double v_ext[3], v_int[3];
        for (int e=0; e < nB; ++e)
        {
            auto compute_exterior_values = (bc(e) == 0) ? absorb : reflect;
            
            for (int i=0; i < n_colloc; ++i)
            {
                v_int[0] = uB(i, 0, 0, e);
                v_int[1] = uB(i, 1, 0, e);
                v_int[2] = uB(i, 2, 0, e);

                compute_exterior_values(v_ext, v_int, &n(0, i, e));

                uB(i, 0, 1, e) = v_ext[0];
                uB(i, 1, 1, e) = v_ext[1];
                uB(i, 2, 1, e) = v_ext[2];
            }
        }

        // flux
        flx->action(uB, uB);

        // add to divF
        prol->t(uB, divF);
    }
} // namespace dg


#endif