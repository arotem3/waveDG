#include "ZeroBoundary.hpp"

namespace dg
{
    void ZeroBoundary::action(int n_var, double * u_) const
    {
        auto u = reshape(u_, n_var, n_dof);

        for (int i : I)
            for (int d = 0; d < n_var; ++d)
                u(d, i) = 0.0;
    }

    ZeroBoundary::ZeroBoundary(const Mesh2D& mesh, const QuadratureRule * basis, int n_faces, const int * faces)
    {
        if (basis->type != QuadratureRule::GaussLobatto)
            wdg_error("ZeroBoundary is only defined for GaussLobatto basis functions.");

        const int n_elem = mesh.n_elem();
        const int n_basis = basis->n;

        n_dof = n_basis * n_basis * n_elem;
        I.reshape(n_basis * n_faces);
        I.fill(-1);

        // map edge index to volume index
        auto E2V = [n_basis](int i, int s, int el) -> int
        {
            const int m = (s == 0 || s == 2) ? i : (s == 1) ? (n_basis-1) : 0;
            const int n = (s == 1 || s == 3) ? i : (s == 2) ? (n_basis-1) : 0;

            return m + n_basis * (n + n_basis * el);
        };

        int l = 0;
        for (int f = 0; f < n_faces; ++f)
        {
            const Edge * edge = mesh.edge(faces[f], FaceType::BOUNDARY);

            const int el = edge->elements[0];
            const int s = edge->sides[0];

            for (int i = 0; i < n_basis; ++i)
            {
                const int idx = E2V(i, s, el);
                I(l) = idx;
                ++l;
            }
        }
    }
} // namespace dg
