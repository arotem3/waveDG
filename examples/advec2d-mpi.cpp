/** @file advec2d-mpi.cpp
 *  @brief Example driver for solving the advection equation with MPI.
 * 
 * This file is a driver for solving the advection equation:
 * $$u_t + \nabla \cdot (\mathbf{c}u) = 0.$$
 * 
 * To compile & run this program, first compile the library in MPI mode:
 * 
 * `cmake . -D WDG_USE_MPI=ON`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make advec2d-mpi`
 * 
 * And run with
 * 
 * `mpirun -np 2 examples/advec2d-mpi`
 * 
 * Adjusting the number of processors as needed.
 * The program will write the collocation points and solution
 * values to `solution/x.%05d` and `solution/u%05d.%05d`, respectively, in binary format.
 * Where the extension is the MPI rank, and the number on u is the time step; e.g. first processor writes:
 * `solution/x.00000` and at time step 10 will write `solution/u00010.00000` and processor 128 writes:
 * `solution/x.00127` and at time step 555 will write `solution/u00555.00127`, etc.
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

/// specify initial condition.
inline static void initial_conditions(const double x[2], double * F)
{
    const double r = 4 * x[0]*x[0] + 100 * x[1]*x[1];
    *F = std::exp(-r);
}

inline static void coefficient(const double x[2], double c[2])
{
    const double r = std::hypot(x[0], x[1]);
    c[0] = r * x[1];
    c[1] = -r * x[0];
}

constexpr static double max_speed()
{
    return M_SQRT2;
}

int main(int argc, char ** argv)
{
    // Initialize MPI environment
    MPIEnv mpi(argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = false;

    // Specify basis functions in terms of 1D quadrature rule. Basis functions
    // are tensor product of 1D Lagrange interpolating polynomials on Gauss
    // quadrature rule. The order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh
    const int nx = 25, ny = 25;
    const double x_min = -1.0, x_max = 1.0, y_min = -1.0, y_max = 1.0;
    Mesh2D mesh;
    if (rank == 0)
        mesh = Mesh2D::uniform_rect(nx, x_min, x_max, ny, y_min, y_max);
    mesh.distribute("rcb");
    
    // mesh statistics
    const int global_n_elem = mesh.global_n_elem(); // all elements in mesh
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_points; // local number of degrees of freedom
    const double h = mesh.min_h(); // shortest length scale

    // Mass Matrix
    MassMatrix<approx_quad> m(1, mesh, basis); // m*u -> (u, v)

    // Project variable coefficient
    MassMatrix<approx_quad> m2(2, mesh, basis);
    LinearFunctional LF2(2, mesh, basis);
    dmat c(2, n_points);
    LF2(coefficient, c);
    m2.inv(c);

    // DG discretization
    Advection<approx_quad> a(1, mesh, basis, c, false); // a*u -> -(c u, grad v) - <(c u)*, v>
    AdvectionHomogeneousBC<approx_quad> bc(1, mesh, basis, c, false); // specify u == 0 outside domain.

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 10.0;

    const double CFL = 0.5 / pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    // this dt is optimal for forward Euler, for higher order we can typically take larger dt
    double dt = CFL / max_speed() * h ;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    if (rank == 0)
    {
        const int global_n_dof = n_colloc * n_colloc * global_n_elem;
        std::cout << "#elements: " << global_n_elem << "\n"
                  << "#DOFs/element: " << n_colloc << "^2\n"
                  << "#DOFs: " << global_n_dof << "\n"
                  << "dx: " << h << "\n"
                  << "dt: " << dt << "\n"
                  << "#times steps: " << nt << "\n";
        if (approx_quad)
            std::cout << "quadrature rule: fast (approximate)\n";
        else
            std::cout << "quadrature rule: exact\n";
    }

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a * u + bc * u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i=0; i < n_dof; ++i)
            dudt[i] = 0.0;
            
        a.action(u, dudt);
        bc.action(u, dudt);
        m.inv(dudt);
    };

    // time integrator
    ode::SSPRK3 rk(n_dof);

    // set up solution vector.
    dcube u(n_colloc, n_colloc, n_elem);

    // initial conditions
    LinearFunctional LF(1, mesh, basis);
    LF(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), 2*n_points, x);
    
    // save initial condition to file
    to_file(std::format("solution/u{:0>5d}.{:0>5d}", 0, rank), n_dof, u);
    
    // Time loop
    std::string progress(30, ' ');
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.{:0>5d}", it/skip, rank), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        if (rank == 0)
            std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    if (rank == 0)
        std::cout << std::endl;
        
    return 0;
}