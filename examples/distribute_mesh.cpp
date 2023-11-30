#include "wavedg.hpp"
#include <fstream>
using namespace dg;

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mesh2D mesh;
    if (rank == 0)
    {
        mesh = Mesh2D::uniform_rect(20, 0.0, 1.0, 20, 0.0, 1.0);
    }

    mesh.distribute(MPI_COMM_WORLD, "rcb");

    auto basis = QuadratureRule::quadrature_rule(2, QuadratureRule::GaussLobatto);

    auto x = mesh.element_physical_coordinates(basis);
    int n_elem = mesh.n_elem();
    int n_colloc = basis->n;
    int n_pts = 2 * n_elem * n_colloc * n_colloc;

    std::ofstream out("solution/x" + std::to_string(rank), std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char*>(x), n_pts * sizeof(double));
    out.close();

    

    MPI_Finalize();
    return 0;
}