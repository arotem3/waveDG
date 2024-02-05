#ifndef WDG_LINALG_HPP
#define WDG_LINALG_HPP

#include <cmath>

#include "Tensor.hpp"

namespace dg
{
    /// @brief computes the (lower) Cholesky decomposition of @a a inplace.
    /// @param[in] m order of matrix @a a
    /// @param[in,out] a shape (m,m). On exit, the lower triangular part of @a a
    /// is set to L such that a = L * L'. The strictly upper triangular elements
    /// are not referenced.
    /// @return true if decomposition is successful, false if failed (i.e.
    /// negative pivot).
    bool chol(int m, double * a);

    /// @brief solves @a x <- @a a \ @a x inplace on @a x, assuming that @a a
    /// containns the lower Cholesky factor of @a a
    /// @param[in] m order of matrix @a a. 
    /// @param[in] a shape (m,m). Lower triangular part contains L such that a = L * L'.
    /// @param[in] n_var number of right hand sides in @a x.
    /// @param[in,out] x shape (n_var, m). On entry, the right hand side of
    /// a*x==b. On exit, x <- a \ x.
    void solve_chol(int m, const double * a, int n_var, double * x);

    /// @brief multiplies @a x <- @a a * @a x inplace on @a x, assuming that @a a
    /// containns the lower Cholesky factor of @a a
    /// @param[in] m order of matrix @a a. 
    /// @param[in] a a shape (m,m). Lower triangular part contains L such that a = L * L'.
    /// @param[in] n_var number of right hand sides in @a x.
    /// @param[in,out] x shape (n_var, m). On exit, x <- a*x.
    void mult_chol(int m, const double * a, int n_var, double * x);

    /// @brief computes the real eigenvalue decomposition of a real matrix if it exists.
    /// @param n rank of a
    /// @param R the eigenvectors
    /// @param e eigenvalues
    /// @param a matrix a to compute eigenvalue decomp for
    /// @return true on successful decomposition. false if failed or if any of
    /// the eigenvalues are complex.
    bool real_eig(int n, double * R, double * e, const double * a);

    /// @brief computes the Euclidean norm of x (with MPI we assume x is
    /// distributed, MPI_Allreduce is called)
    /// @param n length of x
    /// @param x 
    /// @return ||x||
    double norm(int n, const double * x);
    
    /// @brief computes the Euclidean distance between x and y: ||x - y|| (with
    /// MPI we assume x and y are distributed, MPI_Allreduce is called)
    /// @param n length of x and y
    /// @param x 
    /// @param y 
    /// @return ||x - y||
    double error(int n, const double * x, const double * y);


} // namespace dg

#endif