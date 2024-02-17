/** @file advec2d.cpp
 *  @brief Example driver for solving the advection equation
 * 
 * This file is a driver for solving the advection equation:
 * $$u_t + \nabla \cdot (\mathbf{c}u) = 0.$$
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make advec2d`
 * 
 * And run with
 * 
 * `./examples/advec2d`
 * 
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u%05d.00000`, respectively, in binary format.
 * Where the number on u is the time step; e.g. at time step 10 we write `solution/u00010.00000` 
 * and at time step 555 we write `solution/u00555.00000`, etc.
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
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // Specify basis functions in terms of 1D quadrature rule. Basis functions
    // are tensor product of 1D Lagrange interpolating polynomials on Gauss
    // quadrature rule. The order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLobatto;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh
    const int nx = 25, ny = 25;
    Mesh2D mesh = Mesh2D::uniform_rect(nx, -1.0, 1.0, ny, -1.0, 1.0);
    
    // mesh statistics
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_colloc * n_elem; // local total number of collocation points
    const int n_dof = n_points; // local number of degrees of freedom
    const double h = mesh.min_edge_measure(); // shortest length scale

    // Mass Matrix
    MassMatrix<approx_quad> m(1, mesh, basis); // m*u -> (u, v)

    // coefficient c
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

    const double CFL = 1.0 / pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    // this dt is optimal for forward Euler, for higher order we can typically take larger dt
    double dt = CFL / max_speed() * h ;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_colloc << "^2\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

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

    // set up solution vector
    dcube u(n_colloc, n_colloc, n_elem);

    // Project initial conditions
    LinearFunctional LF(1, mesh, basis);
    LF(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", 2*n_points, x);
    
    // save initial condition to file
    to_file(std::format("solution/u{:0>5d}.00000", 0), n_dof, u);
    
    // Time loop
    std::string progress(30, ' '); // progress bar
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u); // time step

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.00000", it/skip), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << std::endl;
    
    return 0;
}