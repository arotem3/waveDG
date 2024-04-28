#include "CGMask.hpp"

namespace dg
{
    void CGMask::mask(int n_var, double * u_) const
    {
        const int n = I.shape(1);
        auto u = reshape(u_, n_var, n_dof);
        
        for (int i = 0; i < n; ++i)
        {
            const int idx = I(0, i);
            for (int d = 0; d < n_var; ++d)
                u(d, idx) = 0.0;
        }
    }

    void CGMask::unmask(int n_var, double * u_) const
    {
        const int n = I.shape(1);
        auto u = reshape(u_, n_var, n_dof);

        for (int i = 0; i < n; ++i)
        {
            const int to = I(0, i);
            const int from = I(1, i);
            for (int d = 0; d < n_var; ++d)
                u(d, to) = u(d, from);
        }
    }

    void CGMask::sum(int n_var, double * u_) const
    {
        const int n = I.shape(1);
        auto u = reshape(u_, n_var, n_dof);

        for (int i = 0; i < n; ++i)
        {
            const int from = I(0, i);
            const int to = I(1, i);
            for (int d = 0; d < n_var; ++d)
                u(d, to) += u(d, from);
        }
    }

    CGMask::CGMask(const Mesh2D& mesh, const QuadratureRule * basis)
        : n_dof(mesh.n_elem() * basis->n * basis->n)
    {
        const int n_basis = basis->n;
        const int n_edges = mesh.n_edges(FaceType::INTERIOR);
        const int n_nodes = mesh.n_nodes();

        // map edge index to volume index
        auto E2V = [n_basis](int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (n_basis-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (n_basis-1) : 0;

            return m + n_basis * (n + n_basis * el);
        };

        // map node to volume index
        auto N2V = [n_basis](int c, int el) -> int
        {
            const int m = (c == 0 || c == 3) ? 0 : (n_basis-1);
            const int n = (c == 0 || c == 1) ? 0 : (n_basis-1);
            
            return m + n_basis * (n + n_basis * el);
        };

        // map redundant DOFs to controlling DOFs
        std::map<int, int> m;

        // iterate over interior edges to indentify duplicates DOFs
        if (n_basis > 2) // <- for n_basis == 2, the degrees of freedom are entirely on the nodes
        {
            for (int e = 0; e < n_edges; ++e)
            {
                auto edge = mesh.edge(e, FaceType::INTERIOR);

                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];

                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];

                const bool reversed = edge->delta < 0;

                for (int i = 1; i < n_basis-1; ++i)
                {
                    const int j = (reversed) ? (n_basis-1-i) : i;

                    const int v0 = E2V(i, s0, el0);
                    const int v1 = E2V(j, s1, el1);
                    m[v1] = v0;
                }
            }
        }

        // iterate over nodes to identify duplicate DOFs
        for (int k = 0; k < n_nodes; ++k)
        {
            auto& node = mesh.node(k);
            
            const int nel = node.connected_elements.size();
            const int el0 = node.connected_elements.at(0).id;
            const int c0 = node.connected_elements.at(0).i;

            const int v0 = N2V(c0, el0);

            for (int i = 1; i < nel; ++i)
            {
                const int el = node.connected_elements.at(i).id;
                const int c = node.connected_elements.at(i).i;

                const int vi = N2V(c, el);
                m[vi] = v0;
            }
        }

        const int n = m.size();
        I.reshape(2, n);
        int l = 0;
        for (auto [to, from] : m)
        {
            I(0, l) = to;
            I(1, l) = from;
            ++l;
        }
    }
} // namespace dg
