#include "CovariantNormalProlongator.hpp"

namespace dg
{
    CovariantNormalProlongator::CovariantNormalProlongator(const Mesh2D& mesh, const QuadratureRule * basis, FaceType face_type_)
        : dim{2},
          n_elem{mesh.n_elem()},
          n_faces{mesh.n_edges(face_type_)},
          n_basis{basis->n},
          face_type{face_type_},
          _v2e(2 * n_basis * n_basis * n_faces),
          P(n_basis)
    {
        if (n_faces == 0)
            return;
        
        const double x[] = {-1.0};
        lagrange_basis_deriv(P, n_basis, basis->x, 1, x);

        _v2e.fill(-1);
        auto v2e = reshape(_v2e, n_basis, n_basis, 2, n_faces);

        const int nc = n_basis;
        auto mapV2E = [nc](int k, int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1-k) : k;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1-k) : k;

            return m + nc * (n + nc * el);
        };

        for (int f = 0; f < n_faces; ++f)
        {
            const Edge * edge = mesh.edge(f, face_type);
            const int el0 = edge->elements[0];
            const int s0 = edge->sides[0];

            for (int i = 0; i < n_basis; ++i)
            {
                for (int k = 0; k < n_basis; ++k)
                {
                    v2e(k, i, 0, f) = mapV2E(k, i, s0, el0);
                }
            }

            if (face_type == FaceType::INTERIOR)
            {
                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                for (int i = 0; i < n_basis; ++i)
                {
                    for (int k = 0; k < n_basis; ++k)
                    {
                        const int j = (edge->delta > 0) ? i : (n_basis-1-i);
                        v2e(k, i, 1, f) = mapV2E(k, j, s1, el1);
                    }
                }
            }
        }
    }

    static void action_2d(int n_elem, int n_edges, int n_basis, int n_var, FaceType face_type, const double * u_, double * uf_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_basis, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_basis * n_basis * n_elem);
        auto v2e = reshape(v2e_, n_basis, n_basis, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        uf.zeros();

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double val = 0.0;
                        for (int k = 0; k < n_basis; ++k)
                        {
                            const int j = v2e(k, i, side, e);
                            val += u(d, j) * P(k);
                        }
                        uf(i, d, side, e) = val;
                    }
                }
            }
        }
    }

    void CovariantNormalProlongator::action(int n_var, const double * u, double * uf) const
    {
        if (n_faces == 0)
            return;

        action_2d(n_elem, n_faces, n_basis, n_var, face_type, u, uf, _v2e, P);
    }

    static void transpose_2d(int n_elem, int n_edges, int n_basis, int n_var, FaceType face_type, const double * uf_, double * u_, const int * v2e_, const dvec& P)
    {
        auto uf = reshape(uf_, n_basis, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_basis * n_basis * n_elem);
        auto v2e = reshape(v2e_, n_basis, n_basis, 2, n_edges);

        const int n_sides = (face_type == FaceType::INTERIOR) ? 2 : 1;

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_basis; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Y = uf(i, d, side, e);
                        for (int k = 0; k < n_basis; ++k)
                        {
                            const int j = v2e(k, i, side, e);
                            u(d, j) += Y * P(k);
                        }
                    }
                }
            }
        }
    }

    void CovariantNormalProlongator::t(int n_var, const double * uf, double * u) const
    {
        if (n_faces == 0)
            return;

        transpose_2d(n_elem, n_faces, n_basis, n_var, face_type, uf, u, _v2e, P);
    }
} // namespace dg
