#include "wavedg.hpp"

using namespace dg;

constexpr bool approx_quad = true;

class Poisson : public Operator
{
public:
    Poisson(const Mesh2D& mesh, const QuadratureRule * basis, int n_faces, const int * dirichlet_faces);

    void action(int n_var, const double * x, double * y) const
    {
        wdg_error("Poisson::action(n_var, x, y) not defined.");
    }

    void action(const double * x, double * y) const;

    /// @brief computes the residual r = b - A*u for some u 
    void residual(const double * U, double * r) const;

    void postprocess(double * U) const
    {
        M.unmask(1, U);
        for (int i=0; i < u.size(); ++i)
            U[i] += G[i];
    }

private:
    const Mesh2D& mesh;
    const QuadratureRule * basis;

    mutable FEMVector G;
    mutable FEMVector u;
    mutable FaceVector uI;
    mutable FaceVector uB;

    CGMask M;
    StiffnessMatrix<approx_quad> S;
    LobattoFaceProlongator I;
    LobattoFaceProlongator B;
    ZeroBoundary BC;
};

void f(const double X[2], double F[1])
{
    // const double x = X[0], y = X[1];
    *F = 1.0;
}

void g(const double X[2], double G[1])
{
    const double x = X[0], y = X[1];
    if (std::abs(x - 1.0) < 1e-12)
        *G = 1.0 - y * y;
    else if (std::abs(x + 1.0) < 1e-12)
        *G = y * (1.0 - y * y);
    else
        *G = 0.0;
}

inline static void to_file(const std::string& fname, int n_dof, const double * u)
{
    std::ofstream out(fname, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(u), n_dof * sizeof(double));
    out.close();
}

int main()
{
    Mesh2D mesh = Mesh2D::uniform_rect(4, -1.0, 1.0, 4, -1.0, 1.0);

    const int n_basis = 4;
    auto basis = QuadratureRule::quadrature_rule(n_basis, QuadratureRule::GaussLobatto);

    const int n_boundary_faces = mesh.n_edges(FaceType::BOUNDARY);
    ivec dirichlet_faces(n_boundary_faces);
    for (int i = 0; i < n_boundary_faces; ++i)
        dirichlet_faces(i) = i;

    Poisson A(mesh, basis, n_boundary_faces, dirichlet_faces);
    FEMVector e(1, mesh, basis);
    FEMVector Ae(1, mesh, basis);
    const int ndof = e.size();
    std::ofstream fout("solution/A.txt");
    fout << std::setprecision(10) << std::scientific;
    for (int i=0; i < ndof; ++i)
    {
        e[i] = 1.0;
        A.action(e, Ae);
        e[i] = 0.0;

        for (int j = 0; j < ndof; ++j)
        {
            fout << std::setw(20) << Ae[j];
        }
        fout << "\n";
    }
    fout.close();

    FEMVector r(1, mesh, basis);
    FEMVector u(1, mesh, basis);
    A.residual(u, r);

    auto out = pcg(ndof, u, &A, r, nullptr, 100, 1e-8, 1);
    A.postprocess(u);

    auto& metrics = mesh.element_metrics(basis);
    const double * x = metrics.physical_coordinates();

    to_file("solution/x.00000", 2 * ndof, x);
    to_file("solution/u.00000", ndof, u);

    return 0;
}

Poisson::Poisson(const Mesh2D& mesh_, const QuadratureRule * basis_, int n_faces_, const int * dirichlet_faces)
    : mesh{mesh_},
      basis{basis_},
      G(1, mesh, basis),
      u(1, mesh, basis),
      uI(1, mesh, FaceType::INTERIOR, basis),
      uB(1, mesh, FaceType::BOUNDARY, basis),
      M(mesh, basis),
      S(mesh, basis),
      I(mesh, basis, FaceType::INTERIOR),
      B(mesh, basis, FaceType::BOUNDARY),
      BC(mesh, basis, n_faces_, dirichlet_faces)
{
    const int n_faces = uB.n_faces();
    const int n_basis = uB.n_basis();

    auto& metrics = mesh.edge_metrics(basis, FaceType::BOUNDARY);
    auto X = reshape(metrics.physical_coordinates(), 2, n_basis, n_faces);

    for (int f = 0; f < n_faces; ++f)
    {
        for (int i = 0; i < n_basis; ++i)
        {
            const double x[] = {X(0, i, f), X(1, i, f)};
            double gi;
            g(x, &gi);
            uB(i, 0, 0, f) = gi;
        }
    }
    B.t(1, uB, G);
}

void Poisson::action(const double * x, double * y) const
{
    copy(u.size(), x, u);
    BC.action(u);
    M.unmask(1, u);

    zeros(u.size(), y);
    S.action(u, y);

    M.sum(1, y);
    M.mask(1, y);
    BC.action(y);
}

void Poisson::residual(const double * U, double * r) const
{
    LinearFunctional2D l(mesh, basis);
    l.action(1, f, r);

    FEMVector v(1, mesh, basis);
    S.action(G, v);

    int n = v.size();
    axpby(n, -1.0, v, 1.0, r);

    M.sum(1, r);
    M.mask(1, r);
    BC.action(r);

    v.as_dvec().zeros();
    S.action(U, v);
    axpby(n, -1.0, v, 1.0, r);
}