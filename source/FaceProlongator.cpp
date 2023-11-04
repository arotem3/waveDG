#include "FaceProlongator.hpp"

namespace dg
{
    LobattoFaceProlongator::LobattoFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type_)
        : n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          edge_type(edge_type_),
          v2e(n_colloc, 2, n_edges)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc](int i, int s, int el) -> int
        {
            const int m = (s == 0 || s == 2) ? i : (s == 1) ? (nc-1) : 0;
            const int n = (s == 1 || s == 3) ? i : (s == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        for (int e = 0; e < n_edges; ++e)
        {
            const Edge * edge = mesh.edge(e, edge_type);
            const int el0 = edge->elements[0];
            const int s0 = edge->sides[0];

            for (int i = 0; i < n_colloc; ++i)
            {
                v2e(i, 0, e) = mapV2E(i, s0, el0);
            }

            if (edge_type == Edge::INTERIOR)
            {
                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = (edge->delta > 0) ? i : (n_colloc - 1 - i);
                    v2e(i, 1, e) = mapV2E(j, s1, el1);
                }
            }
        }
    }

    void LobattoFaceProlongator::action(const double * u_, double * uf_, int n_var) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == Edge::INTERIOR) ? 2 : 1;

        uf.zeros();

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = v2e(i, side, e);

                    for (int d = 0; d < n_var; ++d)
                    {    
                        uf(i, d, side, e) = u(d, j);
                    }
                }
            }
        }
    }

    void LobattoFaceProlongator::t(const double * uf_, double * u_, int n_var) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == Edge::INTERIOR) ? 2 : 1;

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        const int j = v2e(i, side, e);
                        u(d, j) += uf(i, d, side, e);
                    }
                }
            }
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type_)
        : n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          edge_type(edge_type_),
          v2e(n_colloc, n_colloc, 2, n_edges),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        const double x = -1.0;
        lagrange_basis(P, n_colloc, basis->x, 1, &x);

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc](int k, int i, int s, int el) -> int
        {
            const int m = (s == 0 || s == 2) ? i : (s == 1) ? (nc-1-k) : k;
            const int n = (s == 1 || s == 3) ? i : (s == 2) ? (nc-1-k) : k;

            return m + nc * (n + nc * el);
        };

        for (int e = 0; e < n_edges; ++e)
        {
            const Edge * edge = mesh.edge(e, edge_type);
            const int el0 = edge->elements[0];
            const int s0 = edge->sides[0];

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int k = 0; k < n_colloc; ++k)
                {
                    v2e(k, i, 0, e) = mapV2E(k, i, s0, el0);
                }
            }

            if (edge_type == Edge::INTERIOR)
            {
                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = (edge->delta > 0) ? i : (n_colloc-1-i);
                        v2e(k, i, 1, e) = mapV2E(k, j, s1, el1);
                    }
                }
            }
        }
    }

    void LegendreFaceProlongator::action(const double * u_, double * uf_, int n_var) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == Edge::INTERIOR) ? 2 : 1;

        uf.zeros();

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double val = 0.0;
                        for (int k = 0; k < n_colloc; ++k)
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

    void LegendreFaceProlongator::t(const double * uf_, double * u_, int n_var) const
    {
        if (n_edges == 0)
            return;
            
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == Edge::INTERIOR) ? 2 : 1;

        for (int e = 0; e < n_edges; ++e)
        {
            for (int side = 0; side < n_sides; ++side)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        double Y = uf(i, d, side, e);
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            const int j = v2e(k, i, side, e);
                            u(d, j) += Y * P(k);
                        }
                    }
                }
            }
        }
    }

} // namespace dg