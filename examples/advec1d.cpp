/** @file advec1d.cpp
 *  @brief Example driver for solving the 1D advection equation
 * 
 * This file is a driver for solving the advection equation:
 * $$u_t + (\mathbf{c}u)_x = 0.$$
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make advec1d`
 * 
 * And run with
 * 
 * `./examples/advec1d`
 * 
 * The program will write the collocation points and solution
 * values to `solution/x.00000` and `solution/u%05d.00000`, respectively, in binary format.
 * Where the number on u is the time step; e.g. at time step 10 we write `solution/u00010.00000` 
 * and at time step 555 we write `solution/u00555.00000`, etc.
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <format>

#include "wavedg.hpp"

using namespace dg;

inline static void initial_conditions(const double x[], double F[])
{
    *F = std::sin(3 * M_PI * x[0]);
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

inline static void speed(const double x[], double F[])
{
    *F = 1.0 + std::pow(1 - x[0]*x[0], 5);
}

constexpr static double max_speed()
{
    return 2.0;
}

int main(int argc, char ** argv)
{
    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = true;

    // vector dimension of PDE.
    constexpr int n_var = 1;

    // Specify basis functions in terms of quadrature rule. Basis functions are
    // the Lagrange interpolating polynomials on Gauss quadrature rule. The
    // order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 10;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh and specify if mesh should be periodic (connects the last
    // element to the first element)
    const int n_elem = 20;
    const bool periodic = true;
    Mesh1D mesh = Mesh1D::uniform_mesh(n_elem, -1.0, 1.0, periodic);

    // mesh statistics
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_var * n_points;
    const double h = mesh.min_h();

    // Mass Matrix
    MassMatrix<approx_quad> m(mesh, basis);

    // Project variable coefficient
    auto quad = QuadratureRule::quadrature_rule(n_colloc/2 + 5);
    LinearFunctional1D LF(mesh, basis, quad);
    FEMVector c(1, mesh, basis);
    LF.action(1, speed, c);
    m.inv(1, c);

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;
    
    const double CFL = 1.0 / std::pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    double dt = CFL / max_speed() * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "#elements: " << n_elem << "\n"
              << "#DOFs/element: " << n_colloc << "\n"
              << "#DOFs: " << n_dof << "\n"
              << "dx: " << h << "\n"
              << "dt: " << dt << "\n"
              << "#times steps: " << nt << "\n";
    if (approx_quad)
        std::cout << "quadrature rule: fast (approximate)\n";
    else
        std::cout << "quadrature rule: exact\n";

    // DG discretization
    Advection<approx_quad> a(n_var, mesh, basis, c, false);
    AdvectionHomogeneousBC<approx_quad> bc(n_var, mesh, basis, c, false); // if periodic == true, then this does nothing

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a*u + bc*u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;
        
        a.action(u, dudt);
        bc.action(u, dudt);
        m.inv(n_var, dudt);
    };

    // time integrator
    ode::SSPRK3 rk(n_dof);

    // set up solution vector
    FEMVector u(n_var, mesh, basis);

    // Project initial conditions
    LF.action(n_var, initial_conditions, u);
    m.inv(n_var, u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", n_points, x);

    // save initial conditions to file
    to_file(std::format("solution/u{:0>5d}.00000", 0), n_dof, u);

    // Time loop
    std::string progress(30, ' ');
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.00000", it/skip), n_dof, u);

        progress.at(30*(it-1)/nt) = '#';
        std::cout << "[" << progress << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << std::endl;

    return 0;
}