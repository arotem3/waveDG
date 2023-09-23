#include "FaceProjector.hpp"

namespace dg
{
    inline static int vol_to_edges_index_lobatto(int i, int el, int s, const int shape[])
    {
        const int m = (s == 0 || s == 2) ? i : (s == 1) ? (shape[0]-1) : 0;
        const int n = (s == 1 || s == 3) ? i : (s == 2) ? (shape[1]-1) : 0;
        
        return tensor_index(shape, m, n, el);
    }

    inline static int vol_to_edges_index_legendre(int i, int k, int el, int s, const int shape[])
    {
        const int m = (s == 0 || s == 2) ? i : (s == 1) ? (shape[0]-1-k) : k;
        const int n = (s == 1 || s == 3) ? i : (s == 2) ? (shape[1]-1-k) : k;

        return tensor_index(shape, m, n, el);
    }

    FaceProjector::FaceProjector(const Mesh2D& mesh, const QuadratureRule * basis, EdgeType edge_type_)
            : n_elem(mesh.n_elem()), n_edges(mesh.n_edges(edge_type_)), n_colloc(basis->n), basis_type(basis->type), edge_type(edge_type_)
    {
        if (n_edges == 0)
            return;

        _n = mesh.edge_normals(basis, edge_type);

        if (basis_type == QuadratureType::GaussLobatto)
        {
            vol_to_edge.resize(2 * n_colloc * n_edges, -1);
            auto v2e = reshape(vol_to_edge.data(), n_colloc, 2, n_edges);
            const int shape[] = {n_colloc, n_colloc, n_elem};

            for (int e = 0; e < n_edges; ++e)
            {
                const Edge * edge = mesh.edge(e, edge_type);
                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];

                for (int i = 0; i < n_colloc; ++i)
                {
                    v2e(i, 0, e) = vol_to_edges_index_lobatto(i, el0, s0, shape);
                }

                if (edge_type == INTERIOR)
                {
                    const int el1 = edge->elements[1];
                    const int s1 = edge->sides[1];

                    for (int i = 0; i < n_colloc; ++i)
                    {
                        const int j = (edge->delta > 0) ? i : (n_colloc - 1 - i);
                        v2e(i, 1, e) = vol_to_edges_index_lobatto(j, el1, s1, shape);
                    }
                }
            }
        }
        else
        {
            const double x[] = {-1.0};
            V.resize(n_colloc);
            lagrange_basis(V.data(), n_colloc, basis->x, 1, x);

            vol_to_edge.resize(2 * n_colloc * n_colloc * n_edges, -1);
            auto v2e = reshape(vol_to_edge.data(), n_colloc, n_colloc, 2, n_edges);

            const int shape[] = {n_colloc, n_colloc, n_elem};

            for (int e = 0; e < n_edges; ++e)
            {
                const Edge * edge = mesh.edge(e, edge_type);
                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];

                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        v2e(k, i, 0, e) = vol_to_edges_index_legendre(i, k, el0, s0, shape);
                    }
                }

                if (edge_type == INTERIOR)
                {
                    const int el1 = edge->elements[1];
                    const int s1 = edge->sides[1];

                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int k = 0; k < n_colloc; ++k)
                        {
                            const int j = (edge->delta > 0) ? i : (n_colloc-1-i);
                            v2e(k, i, 1, e) = vol_to_edges_index_legendre(j, k, el1, s1, shape);
                        }
                    }
                }
            }
        }
    }

    void FaceProjector::operator()(const double * x_, double * y_, int n_var) const
    {
        if (n_edges == 0)
            return;

        auto y = reshape(y_, n_var, n_colloc, 2, n_edges);
        auto x = reshape(x_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == INTERIOR) ? 2 : 1;

        for (int i=0; i < 2*n_var*n_colloc*n_edges; ++i)
        {
            y_[i] = 0.0;
        }

        if (basis_type == QuadratureType::GaussLobatto)
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, 2, n_edges);
        
            for (int e = 0; e < n_edges; ++e)
            {
                for (int side = 0; side < n_sides; ++side)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            const int j = v2e(i, side, e);
                            y(d, i, side, e) = x(d, j);
                        }
                    }
                }
            }
        }
        else
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, n_colloc, 2, n_edges);

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
                                val += x(d, j) * V[k];
                            }
                            y(d, i, side, e) = val;
                        }
                    }
                }
            }
        }
    }

    void FaceProjector::t(const double * y_, double * x_, int n_var) const
    {
        if (n_edges == 0)
            return;
            
        auto y = reshape(y_, n_var, n_colloc, 2, n_edges);
        auto x = reshape(x_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == INTERIOR) ? 2 : 1;

        if (basis_type == QuadratureType::GaussLobatto)
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, 2, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int side = 0; side < n_sides; ++side)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            const int j = v2e(i, side, e);
                            x(d, j) += y(d, i, side, e);
                        }
                    }
                }
            }
        }
        else
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, n_colloc, 2, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int side = 0; side < n_sides; ++side)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double Y = y(d, i, side, e);
                            for (int k = 0; k < n_colloc; ++k)
                            {
                                const int j = v2e(k, i, side, e);
                                x(d, j) += Y * V[k];
                            }
                        }
                    }
                }
            }
        }
    }

    void FaceProjector::project_normal(const double * x_, double * y_, bool constant_value, int n_var) const
    {
        if (n_edges == 0)
            return;

        auto y = reshape(y_, n_var, n_colloc, 2, n_edges);
        auto x = reshape(x_, n_var, 2, n_colloc * n_colloc * n_elem);
        auto n = reshape(_n, 2, n_colloc, n_edges);

        const int n_sides = (edge_type == INTERIOR) ? 2 : 1;

        for (int i=0; i < y.size(); ++i)
            y[i] = 0.0;

        if (basis_type == QuadratureType::GaussLobatto || constant_value)
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, 2, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int side = 0; side < n_sides; ++side)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        const int j = constant_value ? 0 : v2e(i, side, e);
                        for (int d = 0; d < n_var; ++d)
                        {
                            y(d, i, side, e) = x(d, 0, j) * n(0, i, e) + x(d, 1, j) * n(1, i, e);
                        }
                    }
                }
            }
        }
        else
        {
            auto v2e = reshape(vol_to_edge.data(), n_colloc, n_colloc, 2, n_edges);

            for (int e = 0; e < n_edges; ++e)
            {
                for (int side = 0; side < n_sides; ++side)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            double a0 = 0.0, a1 = 0.0;
                            for (int k = 0; k < n_colloc; ++k)
                            {
                                const int j = v2e(k, i, side, e);
                                a0 += x(d, 0, j) * V[k];
                                a1 += x(d, 1, j) * V[k];
                            }
                            y(d, i, side, e) = a0 * n(0, i, e) + a1 * n(1, i, e);
                        }
                    }
                }
            }
        }
    }
} // namespace dg