/** @file waveholtz-mpi.cpp
 *  @brief Example driver for solving the Helmholtz equation with MPI.
 * 
 * This file is a driver for solving the Helmholtz equation:
 * $$\Delta p + \omega^2 p = f(x).$$
 * For \f$x\in \Omega\subset \R^2\f$ and appropriate boundary conditions.
 * 
 * We solve the Helmholtz equation using the WaveHoltz algorithm via the
 * closely related first order wave equation:
 * $$p_t + \nabla \cdot \mathbf{u} = -\frac{1}{i\omega} f(x) \sin(\omega t)$$
 * $$\mathbf{u}_t + \nabla p = 0.$$
 * 
 * As implemented, WaveHoltz finds a real valued solution, and post processes
 * the solution to find the complex valued to the original problem.
 * 
 * To compile & run this program, first compile the library in MPI mode:
 * 
 * `cmake . -D WDG_USE_MPI=ON`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make waveholtz-mpi`
 * 
 * And run with
 * 
 * `mpirun -np 2 examples/waveholtz-mpi`
 * 
 * Adjusting the number of processors as needed.
 * The program will write the collocation points and solution
 * values to `solution/x.%05d` and `solution/u.%05d`, respectively, in binary format.
 * Where the extension is the MPI rank; e.g. first processor writes:
 * `solution/x.00000` and `solution/u.00000` and processor 128 writes:
 * `solution/x.00127` and `solution/u.00127`, etc.
 */

#include <iostream>
#include <iomanip>
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
    double w = std::pow(10.0 * M_PI, 2);
    double r = std::pow(x[0] + 0.7, 2) + std::pow(x[1] + 0.1, 2);
    F[0] = w/M_PI * std::exp(-w * r);
    F[1] = 0.0;
    F[2] = 0.0;
}

/// @brief specifies the boundary conditions for every edge in `mesh` assuming a the geometry: [-1, 1]x[-1, 1].
/// @param mesh the mesh representing the geometry [-1, 1]x[1, 1].
/// @return integer vector where out[k] is the boundary condition for boundary
/// edge k. If out[k] == 0, then edge k is an approximate absorbing
/// (non-reflecting) boundary, if out[k] == 1, then edge k is a Neumann
/// (reflecting) boundary.
static ivec boundary_conditions(const Mesh2D& mesh)
{
    const int nB = mesh.n_edges(FaceType::BOUNDARY);

    // get edge centers: quadrature rule with collocation point only at center of element
    auto q = QuadratureRule::quadrature_rule(1);

    const double * x_ = mesh.edge_metrics(q, FaceType::BOUNDARY).physical_coordinates();
    auto x = reshape(x_, 2, nB);

    ivec bc(nB);

    for (int e=0; e < nB; ++e)
    {
        const bool left_wall    = std::abs(x(0, e) + 1.0) < 1e-12;
        // const bool right_wall   = std::abs(x(0, e) - 1.0) < 1e-12;
        const bool bottom_wall  = std::abs(x(1, e) + 1.0) < 1e-12;
        // const bool top_wall     = std::abs(x(1, e) - 1.0) < 1e-12;

        if (left_wall || bottom_wall)
            bc(e) = 1; // Neumann (reflecting) boundary condition
        else
            bc(e) = 0; // Absorbing (approximate non-reflecting) boundary condition
    }

    return bc;
}

int main(int argc, char ** argv)
{
    // Initialize MPI environment
    MPIEnv mpi(argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // vector dimension of PDE. In this case 3: (p, u[0], u[1]). We can
    // interpret p as the pressure field and u=(u[0], u[1]) as the velocity (or
    // momentum) field.
    constexpr int n_var = 3;

    // The frequency of the Helmholtz problem.
    const double omega = 10.0;

    // maximum number of WaveHoltz iterations
    const int maxit = 1'000;

    // relative tolerance for WaveHoltz iteration: ||W - Wprev|| / ||Pi(0)|| < tol --> break
    const double tol = 1e-6;

    // Specify basis functions in terms of 1D quadrature rule. Basis functions
    // are tensor product of 1D Lagrange interpolating polynomials on Gauss
    // quadrature rule. The number of collocation points for 1D quadrature rule.
    // The order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh on root and distribute to all processors on MPI_COMM_WORLD
    const int nx = 20, ny = 20;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    mesh.distribute("rcb");

    // mesh statistics
    const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_var * n_points; // local number of degrees of freedom
    const double h = mesh.min_edge_measure(); // shortest length scale

    // Specify boundary conditions
    const ivec bc = boundary_conditions(mesh);

    // initialize WaveHoltz operator
    WaveHoltz WH(omega, mesh, basis, bc, approx_quad);

    // compute the forcing function f(x)
    LinearFunctional L(n_var, mesh, basis);
    dvec f(n_dof);

    L(force, f);
    
    // compute the inhomogeneous part of the WaveHoltz operator pi0 = Pi(0).
    dvec pi0(n_dof);
    WH.pi0(pi0, f);

    const double pi_zero = norm(n_dof, pi0);

    // print summary
    if (rank == 0)
    {
        const int global_n_dof = n_var * n_colloc * n_colloc * global_n_elem;
        std::cout << "#elements: " << global_n_elem << "\n"
                  << "#DOFs/element: " << n_var << " x " << n_colloc << "^2\n"
                  << "#DOFs: " << global_n_dof << "\n"
                  << "dx: " << h << "\n";
        if (approx_quad)
            std::cout << "quadrature rule: fast (approximate)\n";
        else
            std::cout << "quadrature rule: exact\n";
    }

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
        if (rank == 0)
        {
            std::cout << "[" << progress << "]"
                      << std::setw(5) << it << " / " << maxit
                      << " | err = " << std::setw(10) << err
                      << "\r" << std::flush;
        }
        
        if (err < tol)
            break;
    }

    if (rank == 0)
        std::cout << "\nWaveHoltz iteration completed after " << it << " iterations with rel. error ~ " << err << std::endl;

    // postprocess real valued solution to get complex valued solution to Helmholtz equation.
    Wprev = W;
    WH.postprocess(W, Wprev);

    // get collocation points and write to file in binary format.
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), 2*n_points, x);

    // write solution and write to file in binary format.
    to_file(std::format("solution/u.{:0>5d}", rank), 2*n_points, W);
    
    return 0;
}