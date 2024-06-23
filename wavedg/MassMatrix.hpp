#ifndef DG_MASS_MATRIX_HPP
#define DG_MASS_MATRIX_HPP

#include "wdg_config.hpp"
#include "Mesh1D.hpp"
#include "Mesh2D.hpp"
#include "lagrange_interpolation.hpp"
#include "linalg.hpp"
#include "Operator.hpp"


namespace dg
{
    /// @brief Representation of finite element mass matrix.
    ///
    /// @details The mass matrix is defined as:
    /// $$M_{ij} = (\phi_i, \phi_j).$$
    /// Where \f$\{\phi_i\}\f$ are the basis functions.
    ///
    /// Since the basis functions of the DG method are compactly supported on
    /// each element, then the mass matrix for these bases is block diagonal.
    /// Further, if the basis functions are the Lagrange interpolating
    /// polynomials on a set of \f$p\f$ points which define an exact quadrature
    /// rule for polynomials of degree \f$2p-1\f$ (namely, the Gauss-Legendre
    /// points on parallelogram elements), then the basis functions are
    /// orthogonal, thus the mass matrix is diagonal.
    ///
    /// The `MassMatrix` class is specialized to either the block diagonal, or
    /// diagonal case. If @a ApproximateQuadrature == true, then the quadrature
    /// rule inherent on which the basis set is defined is used to approximate
    /// the mass matri which will result in a Diagonal mass matrix, and if
    /// @a ApproximateQuadrature == false, then the entries of the mass matrix
    /// will be computed using an exact quadrature rule which will result in a
    /// block diagonal mass matrix.
    ///
    /// Even if the mass matrix is not truly diagonal, it is common to
    /// approximate the mass matrix by a diagonal matrix. Specifically, if the
    /// basis functions are the Lagrange interpolating polynomials on a set of
    /// \f$p\f$ points for which the quadrature rule is not exact for
    /// polynomials of degree \f$2p-1\f$ (e.g. Gauss-Lobatto which is only
    /// accurate for degree \f$2p-3\f$), then the mass matrix can be
    /// approximated by this quadrature rule which is inexact. With respect to
    /// the inexact quadrature rule, the basis functions are discretely
    /// orthogonal, and the approximate mass matrix will be diagonal. Note that
    /// as \f$p\f$ increases the quadrature error is \f$O(e^{-p})\f$, thus for
    /// high order discretizations, this approximation is justified.
    ///
    /// Ofcourse, the benefit of approximating the mass matrix is that the total
    /// number of operations decreases greatly for high order. specifically, if
    /// \f$M\f$ is the mass matrix and the mesh has \f$n\f$ elements, then
    /// computing \f$Mx\f$ or solving \f$Mx=b\f$ costs \f$O(n p^4)\f$ operations when
    /// \f$M\f$ is block diagonal, and \f$O(n p^2)\f$ when \f$M\f$ is diagonal.
    /// Additionally, the setup cost is \f$O(n p^6)\f$ when \f$M\f$ is block
    /// diagonal, and \f$O(n p^2)\f$ when \f$M\f$ is diagonal. This is because
    /// `MassMatrix<false>` computes and stores the Cholesky factorization of
    /// \f$M\f$ on construction.
    /// @tparam ApproximateQuadrature Specifies if the mass matrix should be
    /// computed exactly or approximated by the quadrature rule of the basis.
    template <bool ApproximateQuadrature>
    class MassMatrix : public InvertibleOperator
    {
    private:
        const int dim;
        const int n_elem;
        const int n_colloc;

        dvec m;

    public:
        /// @brief computes mass matrix associated with (u, v) in two dimensions. If `Diagonal == false`, then also its Cholesky factorization is computed.
        /// @param[in] mesh the two dimensional mesh
        /// @param[in] basis the collocation points for the Lagrange basis set.
        /// @param[in] quad the quadrature rule for computing the integrals:
        /// \f$(\phi_i, \phi_j).\f$ If `Diagonal == true`, then this parameter
        /// is ignored. If `quad == nullptr`, then the quadrature point is
        /// determined by order of basis and elements mapping.
        MassMatrix(const Mesh2D& mesh, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);
        
        /// @brief computes the mass matrix associated with (u, v) in one dimension. If `Diagonal == false`, then also its Cholesky factorization is computed.
        /// @param mesh the one dimensional mesh.
        /// @param basis the collocation points for the Lagrange basis set.
        /// @param quad the quadrature rule for computing the integrals. If
        /// `Diagonal == true`, then `quad` is ignored. If `Diagonal == false`
        /// and `quad == nullptr`, then the quadrature point is the Gauss
        /// Legendre rule with `basis->n` points.
        MassMatrix(const Mesh1D& mesh, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);
        
        ~MassMatrix() = default;

        /// @brief y = M * x (assuming n_var == 1).
        /// @param[in] x shape (n_basis, n_elem). The DG grid function.
        /// @param[out] y shape (n_basis, n_elem). On exit, y <- M * x.
        void action(const double * x, double * y) const override
        {
            action(1, x, y);
        }

        /// @brief y = M * x
        /// @param[in] n_var vector dimension of x and y
        /// @param[in] x shape (n_var, n_basis, n_elem). The DG grid function.
        /// @param[in,out] y shape (n_var, n_basis, n_elem). On exit, y <- M * x.
        void action(int n_var, const double * x, double * y) const override;

        /// @brief Solves M y = x inplace on x, so that on exit x <- M \ x
        /// @param[in,out] x shape (n_basis, n_elem). The DG grid function. On
        /// exit, x <- M \ x.
        void inv(double * x) const override
        {
            inv(1, x);
        }

        /// @brief Solves M * y = x inplace on x, so that on exit, x <- M \ x.
        /// @param[in] n_var 
        /// @param[in,out] x shape (n_var, n_basis, n_elem). On exit, x <- M \ x.
        void inv(int n_var, double * x) const override;
    
        /// @brief Computes the L2 inner-product (x, y)
        /// @param n_var vector dimension of x and y
        /// @param x 
        /// @param y 
        /// @return (x, y)
        double dot(int n_var, const double * x, const double * y) const;

        double dot(const double * x, const double * y) const
        {
            return dot(1, x, y);
        }
    };

    /// @brief Representation of weighted finite element mass matrix.
    ///
    /// @details The mass matrix is defined as:
    /// $$M_{ij} = (A(x) \phi_i, \phi_j).$$
    /// Where \f$\{\phi\}\f$ are the basis functions, \f$A\f$ is the weight
    /// (which is positive definite).
    ///
    /// We can also specify whether A is diagonal. Since the mass matrix will
    /// have the same sparsity pattern as the Kronocker product of A and a
    /// non-weighted mass matrix, so if A is diagonal, we can take advantage of
    /// that structure.
    ///
    /// More details provided by MassMatrix.
    ///
    /// @tparam ApproximateQuadrature Specifies if the mass matrix should be
    /// computed exactly or approximated by the quadrature rule of the basis.
    template <bool ApproximateQuadrature>
    class WeightedMassMatrix : public InvertibleOperator
    {
    private:
        const int n_elem;
        const int n_colloc;
        const int n_var;
        const bool diag_coef;
        
        dvec m;
    
    public:
        /// @brief computes the weighted mass matrix for (A(x)u, v) where A is a
        /// pos. def. matrix. If `Diagonal == false`, then also its Cholesky
        /// factorization is computed.
        /// @param[in] n_var vector dimension of u
        /// @param[in] mesh the mesh
        /// @param[in] A coefficient matrix A(x). Must be pointwise positive
        /// definite. if (A_is_diagonal), shape (n_var, n_colloc, n_colloc, n_elem);
        /// Else, shape (n_var, n_var, n_colloc, n_colloc, n_elem);
        /// @param[in] A_is_diagonal Specify if A is diagonal or full
        /// @param[in] basis the collocation points for the Lagrange basis set.
        /// @param[in] quad the quadrature rule for computing the integrals:
        /// \f$(\phi_i, \phi_j).\f$ If `Diagonal == true`, then this parameter
        /// is ignored. If `quad == nullptr`, then the quadrature point is
        /// determined by order of basis and elements mapping.
        WeightedMassMatrix(int n_var, const Mesh2D& mesh, const double * A, bool A_is_diagonal, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);
        
        WeightedMassMatrix(int n_var, const Mesh1D& mesh, const double * A, bool A_is_diagonal, const QuadratureRule* basis, const QuadratureRule* quad = nullptr)
        {
            // TO DO
            wdg_error("WeightedMassMatrix not yet implemented for 1D meshes.");
        }
        
        ~WeightedMassMatrix() = default;

        /// @brief y = M * x
        /// @param[in] x shape (n_var, n, n, n_elem) where n is the size of the 1D
        /// basis set specified on initialization. The DG grid function.
        /// @param[out] y shape (n_var, n, n, n_elem). On exit, y <- M * x
        void action(const double * x, double * y) const override;

        /// @brief y = M * x
        /// @param[in] n_var IGNORED!! Instead the vector dimension from initialization is used.
        /// @param[in] x (n_var, n, n, n_elem)
        /// @param[in,out] y (n_var, n, n, n_elem)
        void action(int n_var, const double * x, double * y) const override
        {
            action(x, y);
        }

        /// @brief Solves M y = x inplace on x, so that on exit x <- M \ x
        /// @param[in,out] x shape (n_var, n, n, n_elem), where n is the size of the 1D
        /// basis set specified on initialization. The DG grid function. On
        /// exit, x <- M \ x.
        /// @param[in] n_var vector dimension of x
        void inv(double * x) const override;

        /// @brief Solves M y = x inplace on x, so that on exit x <- M \ x
        /// @param[in] n_var IGNORED!! Instead the vector dimension from initialization is used.
        /// @param[in,out] x (n_var, n, n, n_elem)
        void inv(int n_var, double * x) const override
        {
            inv(x);
        }
    };

    /// @brief Mass matrix on the edges of the mesh, in the sense of trace.
    ///
    /// @details On an edge \f$E\f$ the mass matrix is defined:
    /// $$M_{ij} = \langle \phi_i, phi_j \rangle_{E}.$$
    /// Where \f$\{\phi\}\f$ are the basis functions.
    ///
    /// The `EdgeMassMatrix` is mainly used for computing projections onto an
    /// edge.
    ///
    /// For details on the purpose of the `Diagonal` parameter see `MassMatrix`.
    /// @tparam Diagonal specifies if the mass matrix should be diagonal or not.
    template <bool Diagonal>
    class EdgeMassMatrix
    {
    private:
        const int n_edges;
        const int n_colloc;
        std::vector<double> m;
    
    public:
        /// @brief constructs `EdgeMassMatrix`
        /// @param[in] mesh the mesh
        /// @param[in] edge_type interior or boundary edges
        /// @param[in] basis collocation points for the Lagrange basis
        /// @param[in] quad quadrature rule for computing the integrals:
        /// \f$\langle \phi_i, phi_j \rangle_{E}\f$. If `quad == nullptr`
        /// then `quad = basis` is used (since this leads to a diagonal mass
        /// matrix, it is more efficient to use Diagonal=true). If
        /// `Diagonal==true`, then this parameter is ignored.
        EdgeMassMatrix(const Mesh2D& mesh, FaceType edge_type, const QuadratureRule* basis, const QuadratureRule* quad = nullptr);
        ~EdgeMassMatrix() = default;

        /// @brief Computes \f$y = Mx\f$.
        /// @param[in] x input
        /// @param[out] y \f$y = Mx\f$.
        /// @param n_var vector dimension of `x`, so that `x` has shape (n_var, n_colloc, n_colloc, n_elem).
        void action(const double * x, double * y, int n_var=1) const;

        /// @brief Solves \f$Mx = b\f$ inplace on `x`.
        /// @param[in,out] x On entry, `x` is the right hand side \f$b\f$. On exit, `x` is the solution to \f$Mx = b\f$.
        /// @param n_var vector dimension of `x`, so that `x` has shape (n_var, n_colloc, n_colloc, n_elem).
        void inv(double * x, int n_var=1) const;
    };
} // namespace dg

#endif