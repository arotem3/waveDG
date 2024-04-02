/** @file waveholtz1d.cpp
 *  @brief Example driver for solving the 1D Helmholtz equation
 * 
 * This file is a driver for solving the Helmholtz equation:
 * $$u'' + \omega^2 u = f.$$
 * 
 * We solve the Helmholtz equation using the WaveHoltz algorithm via the
 * closely related first order wave equation:
 * $$p_t + \nabla \cdot \mathbf{u} = -\frac{1}{i\omega} f(x) \sin(\omega t)$$
 * $$\mathbf{u}_t + \nabla p = 0.$$
 * 
 * As implemented, WaveHoltz finds a real valued solution, and post processes
 * the solution to find the complex valued to the original problem.
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make waveholtz1d`
 * 
 * And run with
 * 
 * `./examples/waveholtz1d`
 * 
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u.00000`, respectively, in binary format.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>

#include "wavedg.hpp"

using namespace dg;

/// saves solution vector to binary file.
inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

/// The forcing term \f$f(x)\f$ to the Helmholtz equation.
inline static void force(const double x[2], double F[])
{
    double w = 100.0;
    double r = std::pow(x[0] - 0.5, 2);
    F[0] = w/M_PI * std::exp(-w * r);
    F[1] = 0.0;
}

static ivec boundary_conditions(const Mesh1D& mesh)
{
    const int nB = mesh.n_faces(FaceType::BOUNDARY);
    constexpr int REFLECT = 1;
    constexpr int ABSORB = 0;

    ivec bc(nB);
    if (nB == 2)
    {
        bc(0) = REFLECT; // left
        bc(1) = ABSORB; // right
    }
    
    return bc;
}

int main(int argc, char ** argv)
{
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // vector dimension of PDE. In this case 3: (p, u[0], u[1]). We can
    // interpret p as the pressure field and u=(u[0], u[1]) as the velocity (or
    // momentum) field.
    constexpr int n_var = 2;

    // The frequency of the Helmholtz problem.
    const double omega = 10.0;

    // maximum number of WaveHoltz iterations
    const int maxit = 1'000;

    // relative tolerance for WaveHoltz iteration: ||W - Wprev|| / ||Pi(0)|| < tol --> break
    const double tol = 1e-6;

    // Specify basis functions in terms of quadrature rule. Basis functions are
    // the Lagrange interpolating polynomials on Gauss quadrature rule. The
    // order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh and specify if mesh should be periodic (connects the last
    // element to the first element)
    const int n_elem = 100;
    const bool periodic = false;
    Mesh1D mesh = Mesh1D::uniform_mesh(n_elem, -1.0, 1.0, periodic);

    // mesh statistics
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_var * n_points;
    const double h = mesh.min_h();

    // Mass Matrix & projector
    MassMatrix<approx_quad> m(mesh, basis);
    LinearFunctional1D L(mesh, basis);

    // Specify boundary conditions
    const ivec bc = boundary_conditions(mesh);

    // initialize WaveHoltz operator
    WaveHoltz WH(omega, mesh, basis, bc, approx_quad);

    // Forcing function
    dvec f(n_dof);
    L.action(n_var, force, f);

    // compute the inhomogeneous part of the WaveHoltz operator pi0 = Pi(0).
    dvec pi0(n_dof);
    WH.pi0(pi0, f);

    const double pi_zero = norm(n_dof, pi0);

    // print summary
    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

    // initialize solution W = (p, u)
    dvec W(n_dof); // current estimate
    dvec Wprev(n_dof); // previous estimate

    std::string progress(30, ' '); // progress bar
    std::cout << std::setprecision(3) << std::scientific;

    // WaveHoltz iteration
    int it;
    double err;
    for (it = 1; it <= maxit; ++it)
    {
        Wprev = W;

        // u <- S * u + pi0
        WH.S(W);

        for (int i=0; i < n_dof; ++i)
            W(i) += pi0(i);

        err = error(n_dof, W, Wprev) / pi_zero;

        progress.at(30*(it-1)/maxit) = '#';
        std::cout << "[" << progress << "]"
                  << std::setw(5) << it << " / " << maxit
                  << " | err = " << std::setw(10) << err
                  << "\r" << std::flush;
        
        if (err < tol)
            break;
    }
    std::cout << "\nWaveHoltz iteration completed after " << it << " iterations with rel. error ~ " << err << std::endl;

    // postprocess real valued solution to get complex valued solution to
    // Helmholtz equation.
    FEMVector u(2, mesh, basis);
    WH.postprocess(u, W);

    // get collocation points and write to file in binary format.
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", n_points, x);

    // write solution and write to file in binary format.
    to_file("solution/u.00000", 2*n_points, u);
    
    return 0;
}