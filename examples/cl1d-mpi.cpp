/** @file cl1d-mpi.cpp
 *  @brief Example driver for solving the 1D invicid Burger's equation using MPI
 * 
 * This file is a driver for solving the 1D invicid Burger's equation:
 * $$u_t +  \left(\frac{1}{2}u^2\right)_x = 0.$$
 * 
 * To compile & run this program, first compile the library in MPI mode:
 * 
 * `cmake . -D WDG_USE_MPI=ON`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make cl1d-mpi`
 * 
 * And run with
 * 
 * `mpirun -np 2 ./examples/cl1d`
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
#include <fstream>
#include <format>

#include "wavedg.hpp"

using namespace dg;

inline static void initial_conditions(const double x[], double u[])
{
    *u = 1 + 0.5 * std::exp(-30.0 * x[0] * x[0]);
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

// invicid Burger's equation in conservative form
inline static void F(double x, const double u[1], double F[1])
{
    *F = 0.5 * u[0] * u[0];
}

// local Lax-Friedrichs Flux
inline static void LF_flux(double x, const double uL[], const double uR[], double fh[])
{
    double fL, fR;
    F(0, uL, &fL);
    F(0, uR, &fR);

    const double avg = 0.5 * (fL + fR);
    const double c = std::max(std::abs(uL[0]), std::abs(uR[0]));
    const double jmp = uL[0] - uR[0];

    fh[0] = avg + 0.5 * c * jmp;
}

constexpr static double max_speed()
{
    return 1.5;
}

int main(int argc, char ** argv)
{
    // Initialize MPI environment
    MPIEnv env(argc, argv); // calls MPI_Init. On destruction calls MPI_Finalize.

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // approx_quad == true ==> compute integrals using quadrature rule corresponding to the Lagrange basis collocation points.
    // approx_quad == false ==> compute integrals on higher order quadrature rule (automatically determined).
    constexpr bool approx_quad = false;

    // Specify basis functions in terms of quadrature rule. Basis functions are
    // the Lagrange interpolating polynomials on Gauss quadrature rule. The
    // order of the DG discretization is n_colloc - 1/2.
    const int n_colloc = 3;
    QuadratureRule::QuadratureType basis_type = QuadratureRule::GaussLegendre;
    auto basis = QuadratureRule::quadrature_rule(n_colloc, basis_type);

    // construct Mesh.
    const int global_n_elem = 100;
    const bool periodic = true;
    Mesh1D mesh;
    if (rank == 0)
        mesh = Mesh1D::uniform_mesh(global_n_elem, -1.0, 1.0, periodic);
    mesh.distribute();

    // mesh statistics
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_points;
    const double h = mesh.min_h();

    // Mass Matrix
    MassMatrix<approx_quad> m(1, mesh, basis);

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;
    
    const double CFL = 1.0 / std::pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    double dt = CFL / max_speed() * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    if (rank == 0)
    {
        const int global_n_dof = n_colloc * global_n_elem;
        std::cout << "#elements: " << global_n_elem << "\n"
                  << "#DOFs/element: " << n_colloc << "\n"
                  << "#DOFs: " << global_n_dof << "\n"
                  << "dx: " << h << "\n"
                  << "dt: " << dt << "\n"
                  << "#times steps: " << nt << "\n";
        if (approx_quad)
            std::cout << "quadrature rule: fast (approximate)\n";
        else
            std::cout << "quadrature rule: exact\n";
    }

    // DG discretization
    DivF1D<approx_quad> div(1, mesh, basis);
    EdgeFluxF1D flx(1, mesh, FaceType::INTERIOR, basis);

    // map element DOFs to face values (for computing fluxes)
    auto prolongator = make_face_prolongator(1, mesh, basis, FaceType::INTERIOR);
    const int n_faces = mesh.n_faces(FaceType::INTERIOR);
    dmat uI(2, n_faces); // face DOFs for interior faces

    // m * du/dt = a*u + bc*u -> du/dt = m \ (a*u + bc*u).
    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;
        
        div.action(F, u, dudt);

        prolongator->action(u, uI); // compute face values
        flx.action(LF_flux, uI, uI); // compute flux inplace on uI
        
        // need to subtract the flux so:
        for (int i = 0; i < n_faces; ++i)
        {
            uI(0, i) *= -1;
            uI(1, i) *= -1;
        }

        prolongator->t(uI, dudt); // add flux to dudt

        m.inv(dudt);
    };

    // time integrator
    ode::SSPRK3 rk(n_dof);

    // set up solution vector.
    dmat u(n_colloc, n_elem);

    // Project Initial Conditions
    LinearFunctional LF(1, mesh, basis);
    LF(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), n_points, x);

    // save initial conditions to file
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