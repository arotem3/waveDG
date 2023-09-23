#ifndef DG_EIG_HPP
#define DG_EIG_HPP

#include <cmath>
#include <vector>

namespace dg
{
    /// @brief computes the eigenvalue decomposition of a symmetric n by n
    /// matrix so that a = R * diag(e) * R'
    /// @param n rank of a
    /// @param R orthonormal eigenvectors
    /// @param e eigenvalues
    /// @param a matrix to compute eigenvalue decomp of
    /// @return true on successful decomposition
    bool eig(int n, double * R, double * e, const double * a);
} // namespace dg


#endif