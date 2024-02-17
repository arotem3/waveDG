/** @file wave1d-mpi.cpp
 *  @brief Example driver for solving the 1D wave equation with MPI
 * 
 * This file is a driver for solving the wave equation:
 * $$p_t + u_x = f,$$
 * $$u_t + p_x = g.$$
 * 
 * To compile & run this program, first compile the library in MPI mode:
 * 
 * `cmake . -D WDG_USE_MPI=ON`
 * `make wavedg -j`
 * 
 * Then compile this file with
 * 
 * `make wave1d-mpi`
 * 
 * And run with
 * 
 * `mpirun -np 2 ./examples/wave1d-mpi`
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

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
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
    for (int f = 0; f < nB; ++f)
    {
        auto& face = mesh.face(f, FaceType::BOUNDARY);
        if (face.elements[0] < 0) // left boundary
            bc(f) = REFLECT;
        else // right boundary
            bc(f) = ABSORB;
    }
    
    return bc;
}

int main(int argc, char ** argv)
{
    // Initialize MPI environment
    MPIEnv env(argc, argv); // calls MPI_Init. On destruction calls MPI_Finalize.

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    const int global_n_elem = 100;
    const bool periodic = false;
    Mesh1D mesh;
    if (rank == 0)
        mesh = Mesh1D::uniform_mesh(global_n_elem, -1.0, 1.0, periodic);
    mesh.distribute();

    // mesh statistics
    const int n_elem = mesh.n_elem(); // elements on processor
    const int n_points = n_colloc * n_elem;
    const int n_dof = n_var * n_points;
    const double h = mesh.min_h();

    // Mass Matrix & projector
    MassMatrix<approx_quad> m(n_var, mesh, basis);
    LinearFunctional L(n_var, mesh, basis);

    // time interval: [0, T]
    double t = 0.0; // time variable
    const double T = 2.0;

    const double CFL = 1.0 / std::pow(n_colloc, 2); // Courant-Friedrich-Levy constant

    double dt = CFL / max_speed() * h;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    if (rank == 0)
    {
        const int global_n_dof = n_var * n_colloc * global_n_elem;
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

    // DG Discretization
    WaveEquation a(mesh, basis, approx_quad);
    
    // Boundary conditions
    const ivec _bc = boundary_conditions(mesh);
    WaveBC bc(mesh, _bc, basis);

    // Forcing term
    dvec f(n_dof);

    auto time_derivative = [&](double * dudt, const double t, const double * u) -> void
    {
        for (int i = 0; i < n_dof; ++i)
            dudt[i] = 0.0;

        a.action(u, dudt);
        bc.action(u, dudt);

        L([t](const double * x_, double * f_) -> void {force(t, x_, f_);}, f);
        for (int i=0; i < n_dof; ++i)
            dudt[i] += f(i);
        
        m.inv(dudt);
    };

    // time integrator
    ode::RungeKutta2 rk(n_dof);

    // set up solution vector.
    dcube u(n_var, n_colloc, n_elem);

    // initial conditions
    L(initial_conditions, u);
    m.inv(u);

    // save solution collocation points to file
    auto x = mesh.element_metrics(basis).physical_coordinates();
    to_file(std::format("solution/x.{:0>5d}", rank), n_points, x);

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