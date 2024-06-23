#include "wavedg.hpp"
#include "examples.hpp"
#include "INS.hpp"
#include <format>

static void u_bc(const double X[2], double u[2])
{
    constexpr double C = -6.0 / (0.41 * 0.41);
    const double x = X[0], y = X[1];

    if (std::abs(x + 0.2) < 1e-12)
        u[0] = C * (y - 0.21) * (y + 0.2);
    else
        u[0] = 0.0;
    u[1] = 0.0;
}

static double u_bc_time_ramp(double t)
{
    if (t <= 4)
        return std::sin(M_PI * t / 8.0);
    else
        return 1.0;
}

static double u_bc_time_deriv(double t)
{
    if (t <= 4)
        return M_PI/8.0 * std::cos(M_PI * t / 8.0);
    else
        return 0.0;
}

static ivec outflow_boundaries(const Mesh2D& mesh)
{
    std::vector<int> b;
    
    double X[2];

    const int nf = mesh.n_edges(FaceType::BOUNDARY);
    for (int f = 0; f < nf; ++f)
    {
        const Edge * edge = mesh.edge(f, FaceType::BOUNDARY);
        
        edge->physical_coordinates(0.0, X);
        const double x = X[0], y = X[1];

        if (std::abs(x - 2.0) < 1e-12)
            b.push_back(f);
    }

    int n_outflow = b.size();
    ivec outflows(n_outflow);
    for (int i=0; i < n_outflow; ++i)
        outflows[i] = b.at(i);

    return outflows;
}

static ivec not_outflow(const Mesh2D& mesh)
{
    ivec outflow_ = outflow_boundaries(mesh);
    std::set<int> outflow;
    for (int f : outflow_)
        outflow.insert(f);

    int nf = mesh.n_edges(FaceType::BOUNDARY);
    ivec F(nf - outflow_.size());
    int l = 0;
    for (int f=0; f < nf; ++f)
    {
        if (not outflow.contains(f))
        {
            F[l] = f;
            ++l;
        }
    }

    return F;
}

static void bc_extension(const Mesh2D& mesh, const QuadratureRule * basis, const LobattoFaceProlongator& B, double * G)
{
    FaceVector g(2, mesh, FaceType::BOUNDARY, basis);

    const int n_basis = basis->n;
    const int n_edges = mesh.n_edges(FaceType::BOUNDARY);
    auto& metrics = mesh.edge_metrics(basis, FaceType::BOUNDARY);
    auto X = reshape(metrics.physical_coordinates(), 2, n_basis, n_edges);

    for (int f = 0; f < n_edges; ++f)
    {
        for (int i = 0; i < n_basis; ++i)
        {
            const double x[] = {X(0, i, f), X(1, i, f)};
            double gi[2];

            u_bc(x, gi);

            g(i, 0, 0, f) = gi[0];
            g(i, 1, 0, f) = gi[1];
        }
    }

    B.t(2, g, G);
}

/// @brief nonlinear term u*u'
inline static void F(const double x[2], const double u[2], double f[4])
{
    f[0] = u[0]*u[0];
    f[1] = u[0]*u[1];
    f[2] = f[1];
    f[3] = u[1]*u[1];
}

/// @brief rusonov numerical flux
inline static void rusonov(const double x[2], const double n[2], const double uL[2], const double uR[2], double fh[2])
{
    double fL[4], fR[4];
    F(x, uL, fL);
    F(x, uR, fR);

    double c = 2.0 * std::max(std::hypot(uL[0],uL[1]), std::hypot(uR[0], uR[1]));

    for (int d = 0; d < 2; ++d)
        fh[d] = 0.5 * n[0] * (fL[d] + fR[d]) + 0.5 * n[1] * (fL[2+d] + fR[2+d]) + 0.5 * c * (uL[d] - uR[d]);
}

int main()
{
    const double tol = 1e-10; // tolerance for linear solves
    const double nu = 1e-3; // 1 / Re
    const double T = 10.0;

    Mesh2D mesh = Mesh2D::from_file("meshes/Cyl");
    const double h = mesh.min_h();
    
    const int n_basis = 6;
    auto basis = QuadratureRule::quadrature_rule(n_basis, QuadratureRule::GaussLobatto);
    auto quad = QuadratureRule::quadrature_rule(3*n_basis/2 + 1);

    FEMVector p(1, mesh, basis); // pressure
    FEMVector u(2, mesh, basis); // velocity

    // work variables
    FEMVector u_prev(2, mesh, basis), u_tilde(2, mesh, basis);
    FEMVector N0(2, mesh, basis), N1(2, mesh, basis), GradP(2, mesh, basis);
    FEMVector w(1, mesh, basis), curl_w(2, mesh, basis); // vorticity
    FEMVector DivU(1, mesh, basis);
    FEMVector G(2, mesh, basis); // extension of dirichlet boundary conditions
    
    FaceVector p_neumann(1, mesh, FaceType::BOUNDARY, basis); // pressure Neumann boundary conditions
    FaceVector pn1(1, mesh, FaceType::BOUNDARY, basis), pn0(1, mesh, FaceType::BOUNDARY, basis);

    const int u_dof = u.size();
    const int p_dof = p.size();

    // Set up operators
    DivF2D<false> N(mesh, basis, quad); // div(u*u)
    StiffnessMatrix<true> S(mesh, basis); // (grad u, grad phi)
    MassMatrix<true> M(mesh, basis); // (u, phi)

    FEMVector _m(1, mesh, basis);
    _m.as_dvec().ones();
    M.action(_m, _m);
    to_file("solution/mass.00000", _m.size(), _m);

    LobattoFaceProlongator I(mesh, basis, FaceType::INTERIOR);
    FaceVector uI(2, mesh, FaceType::INTERIOR, basis);
    EdgeFluxF2D<false> flxI(mesh, FaceType::INTERIOR, basis, quad);

    LobattoFaceProlongator B(mesh, basis, FaceType::BOUNDARY);
    FaceVector uB(2, mesh, FaceType::BOUNDARY, basis);
    EdgeFluxF2D<false> flxB(mesh, FaceType::BOUNDARY, basis, quad);
    FaceVector gB(2, mesh, FaceType::BOUNDARY,  basis);

    Nabla<true> D(mesh, basis); // first derivative operations

    CGMask cgm(mesh, basis);
    DG2CG Pr(cgm, mesh, basis);
    
    ivec outflows = outflow_boundaries(mesh);
    ZeroBoundary Zp(mesh, basis, outflows.size(), outflows);

    const int n_boundary_faces = mesh.n_edges(FaceType::BOUNDARY);
    ivec dirichlet_boundaries = not_outflow(mesh);
    const int n_dirichlet_faces = dirichlet_boundaries.size();
    ZeroBoundary Zu(mesh, basis, n_dirichlet_faces, dirichlet_boundaries);

    PressureElliptic PressOp(p_dof, &S, cgm, Zp);
    VelocityElliptic VelOp(u_dof, &M, &S, cgm, Zu);

    auto& fmetrics = mesh.edge_metrics(basis, FaceType::BOUNDARY);
    auto normals = reshape(fmetrics.normals(), 2, n_basis, n_boundary_faces);
    auto detJf = reshape(fmetrics.measures(), n_basis, n_boundary_faces);

    auto& metrics = mesh.element_metrics(basis);
    auto X = metrics.physical_coordinates();
    to_file("solution/x.00000", u_dof, X);
    
    auto W = reshape(basis->w, n_basis);

    // compute BC
    bc_extension(mesh, basis, B, G);
    B.action(2, G, gB);

    // time stepping parameters
    constexpr double gamma0 = 1.5, alpha0 = 2.0, beta0 = 2.0, alpha1 = -0.5, beta1 = -1.0;
    
    double t = 0.0;
    double dt = h / n_basis / n_basis;
    const int nt = std::ceil(T / dt);
    dt = T / nt;

    std::cout << "Solving Incompressible Navier Stokes for t < " << T << "\n"
              << "\t#elements = " << mesh.n_elem() << "\n"
              << "\tpolynomial degree = " << basis->n - 1 << "\n"
              << "\tmin h = " << mesh.min_h() << "\n"
              << "\tmax h = " << mesh.max_h() << "\n"
              << "\tdt = " << dt << "\n"
              << "\tNt = " << nt << "\n"
              << "\t#DOFs u = " << u_dof << "\n"
              << "\t#DOFs p = " << p_dof << "\n";

    int skip = 50;
    std::cout << std::setprecision(4);
    ProgressBar progress_bar(nt);
    int snapshot = 0;
    for (int it = 0; it < nt; ++it)
    {
        // compute nonlinear term: N(u) = -(F, grad phi) + <F.n, phi>
        zeros(u_dof, N0);
        N.action(2, F, u, N0);
        for (int i=0; i < u_dof; ++i)
            N0[i] *= -1.0;
        
        I.action(2, u, uI);
        flxI.action(2, rusonov, uI, uI);
        I.t(2, uI, N0);
        
        B.action(2, u, uB);
        double U = u_bc_time_ramp(t);
        for (int f : dirichlet_boundaries) // set exterior values to Dirichlet values
        {
            for (int i=0; i < n_basis; ++i)
            {
                uB(i, 0, 1, f) = U * gB(i, 0, 0, f);
                uB(i, 1, 1, f) = U * gB(i, 1, 0, f);
            }
        }
        for (int f : outflows) // set exterior values to interior values at outflow
        {
            for (int i=0; i < n_basis; ++i)
            {
                uB(i, 0, 1, f) = uB(i, 0, 0, f);
                uB(i, 1, 1, f) = uB(i, 1, 0, f);
            }
        }
        flxB.action(2, rusonov, uB, uB);
        B.t(2, uB, N0);

        M.inv(2, N0);

        // compute u~: (gamma0 u~ - alpha0 u - alpha1 u_prev) / dt = -beta0 N(u) - beta1 N(u_prev)
        for (int i=0; i < u_dof; ++i)
            u_tilde[i] = (alpha0/gamma0) * u[i] + (alpha1/gamma0) * u_prev[i] - (dt/gamma0)*(beta0*N0[i] + beta1*N1[i]);

        // setup pressure Poisson solve
        D.div(u_tilde, DivU);
        for (int i=0; i < p_dof; ++i)
            DivU[i] *= -gamma0/dt;

        D.xycurl(u, w);
        M.inv(1, w);
        D.zcurl(w, curl_w);
        M.inv(2, curl_w);

        B.action(2, curl_w, uB);
        for (int f = 0; f < n_boundary_faces; ++f)
            for (int i = 0; i < n_basis; ++i)
                pn0(i,0,0,f) = nu * (normals(0,i,f) * uB(i,0,0,f) + normals(1,i,f) * uB(i,1,0,f));

        double dudt = u_bc_time_deriv(t);
        for (int f = 0; f < n_boundary_faces; ++f)
            for (int i = 0; i < n_basis; ++i)
                pn0(i,0,0,f) += dudt * (normals(0,i,f) * gB(i,0,0,f) + normals(1,i,f) * gB(i,1,0,f));

        B.action(2, N0, uB);
        for (int f = 0; f < n_boundary_faces; ++f)
            for (int i = 0; i < n_basis; ++i)
                pn0(i,0,0,f) += normals(0,i,f) * uB(i,0,0,f) + normals(1,i,f) * uB(i,1,0,f);

        for (int f = 0; f < n_boundary_faces; ++f)
        {
            for (int i = 0; i < n_basis; ++i)
            {
                double dpdn = -beta0 * pn0(i,0,0,f) - beta1*pn1(i,0,0,f);
                p_neumann(i,0,0,f) = dpdn * W(i) * detJf(i, f);
            }
        }

        B.t(1, p_neumann, DivU); // DivU <- -gamma0/dt * (div u, phi) + <dp/dn, phi>

        PressOp.residual(p, DivU); // DivU <- DivU - (grad p, grad phi) initial residual for pressure elliptic solve
        cgm.mask(1, p);
        if (norm(p_dof, DivU) > 1e-12)
            auto p_solve_out = pcg(p_dof, p, &PressOp, DivU, nullptr, p_dof, tol);
        PressOp.postprocess(p); // make continuous

        // compute u~~, gamma0 (u~~ - u~)/dt = -grad p
        D.grad(p, GradP);
        M.inv(2, GradP);

        for (int i=0; i < u_dof; ++i)
        {
            u_tilde[i] -= (dt/gamma0) * GradP[i];
        }
        
        M.action(2, u_tilde, u_tilde);

        // copy u to u_prev
        copy(u_dof, u, u_prev);
        copy(u_dof, N0, N1);
        copy(pn0.size(), pn0, pn1);

        // solve for u at next time step
        VelOp.set_alpha(gamma0 / dt / nu);
        U = u_bc_time_ramp(t+dt);
        VelOp.prepare(U, G, u, u_tilde);
        if (norm(u_dof, u_tilde) > 1e-12)
            auto u_solve_out = pcg(u_dof, u, &VelOp, u_tilde, nullptr, u_dof, tol);
        VelOp.post_process(U, G, u);

        t += dt;

        double pnorm = M.dot(p,p);
        double unorm = M.dot(2,u,u);
        ++progress_bar;

        std::cout << "[" << progress_bar.get() << "]  |  "
                  << std::setw(10) << it << " / " << nt << "  |  "
                  << "t = " << std::setw(10) << std::fixed << t << "  |  "
                  << "norm(p) = " << std::setw(10) << std::scientific << pnorm << "  |  "
                  << "norm(u) = " << std::setw(10) << std::scientific << unorm << "\r" << std::flush;

        if (t >= 5 && it % skip == 0)
        {
            to_file(std::format("solution/u.{:0>5d}", snapshot), u_dof, u);
            ++snapshot;
        }
    }
    printf("\n");

    return 0;
}