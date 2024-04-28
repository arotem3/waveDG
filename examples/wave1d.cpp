/** @file wave1d.cpp
 *  @brief Example driver for solving the 1D wave equation
 * 
 * This file is a driver for solving the wave equation:
 * $$p_t + u_x = f,$$
 * $$u_t + p_x = g.$$
 * 
 * To compile & run this program, first compile the library in serial mode:
 * 
 * `cmake . -D WDG_USE_MPI=OFF`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make wave1d`
 * 
 * And run with
 * 
 * `./examples/wave1d`
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
#include "examples.hpp"

using namespace dg;

inline static void initial_conditions(const double x[], double F[])
{
    F[0] = 0.0;
    F[1] = 0.0;
}

inline static void force(const double t, const double x[2], double F[])
{
    const double r = x[0] * x[0];
    F[0] = 10.0 * std::exp(-100.0 * r) * std::sin(30 * t);
    F[1] = 0.0;
}

constexpr static double max_speed()
{
    return 1.0;
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

    // vector dimension of PDE. In this case 2: (p, u).
    constexpr int n_var = 2;

    // Specify basis functions in terms of quadrature rule. Basis functions are
    // the Lagrange interpolating polynomials on Gauss quadrature rule. The
    // order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 5;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh and specify if mesh should be periodic (connects the last
    // element to the first element)
    const int n_elem = 20;
    const bool periodic = false;
    Mesh1D mesh = Mesh1D::uniform_mesh(n_elem, -1.0, 1.0, periodic);

    // mesh statistics
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_var * n_points;
    const double h = mesh.min_h();

    // Mass Matrix & projector
    MassMatrix<approx_quad> m(mesh, basis);
    LinearFunctional1D L(mesh, basis);

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

    // DG Discretization
    WaveEquation a(mesh, basis, approx_quad);
    
    // Boundary conditions
    const ivec _bc = boundary_conditions(mesh);
    WaveBC bc(mesh, _bc, basis);

    // Forcing term
    FEMVector f(n_var, mesh, basis);
    auto F = reshape(f.get(), f.size());

    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;

        a.action(u, dudt);
        bc.action(u, dudt);

        L.action(n_var, [t](const double * x_, double * f_) -> void {force(t, x_, f_);}, F);
        for (int i=0; i < n_dof; ++i)
            dudt[i] += F(i);
        
        m.inv(n_var, dudt);
    };

    // time integrator
    ode::RungeKutta2 rk(n_dof);

    // set up solution vector.
    FEMVector u(n_var, mesh, basis);

    // initial conditions
    L.action(n_var, initial_conditions, u);
    m.inv(n_var, u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file("solution/x.00000", n_points, x);

    // save initial condition to file
    to_file(std::format("solution/u{:0>5d}.00000", 0), n_dof, u);

    // Time loop
    ProgressBar progress_bar(nt);
    constexpr int skip = 10; // save solution every skip time steps
    for (int it = 1; it <= nt; ++it)
    {
        rk.step(dt, time_derivative, t, u);

        if (it % skip == 0)
            to_file(std::format("solution/u{:0>5d}.00000", it/skip), n_dof, u);

        ++progress_bar;
        std::cout << "[" << progress_bar.get() << "]" << std::setw(5) << it << " / " << nt << "\r" << std::flush;
    }
    std::cout << std::endl;

    return 0;
}