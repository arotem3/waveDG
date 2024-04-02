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
    class WaveEquation : public Operator
    {
    public:
        /// @brief initialize DG discretization of wave equation
        /// @param mesh the 2d mesh
        /// @param basis basis function
        /// @param approx_quad whether to use approximate quadrature (fast quadrature) or exact quadrature
        /// @param quad quadrature rule. Only referenced f approx_quad == false.
        /// If quad == nullptr, a quadrature rule is automatically selected.
        WaveEquation(const Mesh2D& mesh, const QuadratureRule * basis, bool approx_quad=false, const QuadratureRule * quad=nullptr);

        /// @brief initialize DG discretization of wave equation
        /// @param mesh the 1d mesh
        /// @param basis basis function
        /// @param approx_quad whether to use approximate quadrature (fast quadrature) or exact quadrature
        /// @param quad quadrature rule. Only referenced f approx_quad == false.
        /// If quad == nullptr, a quadrature rule is automatically selected.
        WaveEquation(const Mesh1D& mesh, const QuadratureRule * basis, bool approx_quad=false, const QuadratureRule * quad=nullptr);

        ~WaveEquation() = default;

        /// @brief Dw <- [(u, grad phi) - <n.u*, phi>, (p, grad phi) - <n p*, phi>]
        /// @param[in] w wave equation discretization: w = [p, u]. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Dw Shape (3, n_colloc, n_colloc, n_elem)
        void action(const double * w, double * Dw) const override;

        /// @brief Dw <- [(u, grad phi) - <n.u*, phi>, (p, grad phi) - <n p*, phi>]
        /// @param[in] n_var IGNORED
        /// @param[in] w wave equation discretization: w = [p, u]. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Dw Shape (3, n_colloc, n_colloc, n_elem)
        void action(int n_var, const double * w, double * Dw) const override
        {
            action(w, Dw);
        }

    private:
        const int dim;

        std::unique_ptr<Operator> div;
        std::unique_ptr<Operator> flx;
        std::unique_ptr<FaceProlongator> prol;

        mutable FaceVector uI;
    };

    inline void reflect_2d(double v_ext[3], const double v_int[3], const double n[2])
    {
        v_ext[0] = v_int[0];
        v_ext[1] = (n[1]*n[1] - n[0]*n[0]) * v_int[1] - 2.0*n[0]*n[1] * v_int[2];
        v_ext[2] = -2.0*n[0]*n[1] * v_int[1] + (n[0]*n[0] - n[1]*n[1]) * v_int[2];
    }

    inline void reflect_1d(double v_ext[2], const double v_int[2], double n)
    {
        v_ext[0] =  v_int[0];
        v_ext[1] = -v_int[1];
    }

    inline void absorb_2d(double v_ext[3], const double v_int[3], const double n[2])
    {
        const double outgoing = v_int[0] + n[0] * v_int[1] + n[1] * v_int[2];
        const double tangential_velocity = n[1] * v_int[1] - n[0] * v_int[2];
        v_ext[0] =  n[0] * v_ext[1] + n[1] * v_ext[2];
        v_ext[1] =  n[1] * tangential_velocity + 0.5 * n[0] * outgoing;
        v_ext[2] = -n[0] * tangential_velocity + 0.5 * n[1] * outgoing;
    }

    inline void absorb_1d(double v_ext[2], const double v_int[2], double n)
    {
        v_ext[0] = 0.5 * (n * v_int[0] + v_int[1]);
        v_ext[1] = 0.5 * (v_int[0] + n * v_int[1]);
    }

    class WaveBC : public Operator
    {
    public:
        /// @brief wave equation boundary conditions
        /// @param mesh 2d mesh
        /// @param bc Specifies the boundary condition on each boundary edge as either absorbing (0) or reflecting/Neumann (1).
        /// @param basis collocation points of Lagrange basis functions
        /// @param approx_quad whether to use approximate quadrature (fast quadrature) or exact quadrature
        /// @param quad quadrature rule. Only referenced f approx_quad == false.
        /// If quad == nullptr, a quadrature rule is automatically selected.
        WaveBC(const Mesh2D& mesh, const int * bc, const QuadratureRule * basis, bool approx_quad, const QuadratureRule * quad=nullptr);

        /// @brief wave equation boundary conditions
        /// @param mesh 2d mesh
        /// @param bc Specifies the boundary condition on each boundary edge as either absorbing (0) or reflecting/Neumann (1).
        /// @param basis collocation points of Lagrange basis functions
        WaveBC(const Mesh1D& mesh, const int * bc, const QuadratureRule * basis);

        ~WaveBC() = default;

        /// @brief applies boundary conditions Bw <- B(w)
        /// @param[in] w solution vector. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Bw output. Shape (3, n_colloc, n_colloc, n_elem)
        void action(const double * w, double * Bw) const override;

        /// @brief applies boundary conditions Bw <- B(w)
        /// @param[in] n_var IGNORED
        /// @param[in] w solution vector. Shape (3, n_colloc, n_colloc, n_elem)
        /// @param[out] Bw output. Shape (3, n_colloc, n_colloc, n_elem)
        void action(int n_var, const double * w, double * Bw) const override
        {
            action(w, Bw);
        }

    private:
        const int dim;
        const int nB;
        const int n_colloc;
        
        std::unique_ptr<Operator> flx;
        std::unique_ptr<FaceProlongator> prol;
        
        const_dcube_wrapper normals_2d;
        dvec normals_1d;
        ivec bc;
        
        mutable FaceVector uB;
    };
} // namespace dg


#endif