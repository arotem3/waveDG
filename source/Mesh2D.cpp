#include "Mesh2D.hpp"

namespace dg
{
    Mesh2D Mesh2D::from_vertices(int nx, const double * x_, int nel, const int * elems_)
    {
        auto coo = reshape(x_, 2, nx);
        auto elems = reshape(elems_, 4, nel);

        constexpr int emap1[] = {0,1,3,0};
        constexpr int emap2[] = {1,2,2,3};

        Mesh2D mesh;
        mesh._elements.resize(nel);
        
        dmat x(2, 4);

        // construct elements
        for (int el = 0; el < nel; ++el)
        {
            for (int i = 0; i < 4; ++i)
            {
                int c = elems(i, el);
                x(0, i) = coo(0, c);
                x(1, i) = coo(1, c);
            }

            mesh._elements[el].reset(new QuadElement(x.data()));
            mesh._elements[el]->id = el;
        }

        // construct edges
        std::unordered_map<int, int> edge_map;
        auto key = [nx](int i, int j) -> int
        {
            return std::min(i, j) + nx * std::max(i, j);
        };
        int edge_id = 0;

        for (int el = 0; el < nel; ++el)
        {
            for (int s = 0; s < 4; ++s)
            {
                const int l1 = emap1[s];
                const int l2 = emap2[s];

                const int C0 = elems(l1, el);
                const int C1 = elems(l2, el);

                const int k = key(C0, C1);
                if (not edge_map.contains(k))
                {
                    const double * x0 = x_ + 2*C0;
                    const double * x1 = x_ + 2*C1;
                    mesh._edges.push_back(std::unique_ptr<Edge>(new StraightEdge(x0, x1, s)));
                    Edge * edge = mesh._edges[edge_id].get();

                    edge->elements[0] = el;
                    edge->id = edge_id;
                    edge->sides[0] = s;
                    edge->type = BOUNDARY;
                    edge->delta = 1;
                    edge_map[k] = edge_id++;
                }
                else
                {
                    int e = edge_map.at(k);
                    Edge * edge = mesh._edges[e].get();

                    int e0 = edge->elements[0];
                    int s0 = edge->sides[0];
                    int n1 = elems(emap1[s0], e0);

                    edge->elements[1] = el;
                    edge->sides[1] = s;
                    edge->type = INTERIOR;
                    edge->delta = (C0 == n1) ? 1 : -1;
                }
            }
        }

        // boundary edges
        for (const auto& edge : mesh._edges)
        {
            if (edge->type == BOUNDARY)
                mesh._boundary_edges.push_back(edge->id);
            else
                mesh._interior_edges.push_back(edge->id);
        }

        return mesh;
    }

    #define ELEMENT_METRICS(name, fun, map, dim)                                                                        \
    const double * Mesh2D::element_ ##name (const QuadratureRule * quad) const                                          \
    {                                                                                                                   \
        if (not map.contains(quad))                                                                                     \
        {                                                                                                               \
            const int m = quad->n;                                                                                      \
            const int nel = n_elem();                                                                                   \
            double * ptr = map.insert({quad, std::unique_ptr<double[]>(new double[dim*m*m*nel])}).first->second.get();  \
                                                                                                                        \
            const int sizes[] = {dim, m, m, nel};                                                                       \
            double xi[2];                                                                                               \
                                                                                                                        \
            for (int el = 0; el < nel; ++el)                                                                            \
            {                                                                                                           \
                const Element * elem = element(el);                                                                     \
                for (int j = 0; j < m; ++j)                                                                             \
                {                                                                                                       \
                    xi[1] = quad->x[j];                                                                                 \
                    for (int i = 0; i < m; ++i)                                                                         \
                    {                                                                                                   \
                        xi[0] = quad->x[i];                                                                             \
                        double * metric = ptr + tensor_index(sizes, 0, i, j, el);                                       \
                        fun;                                                                                            \
                    }                                                                                                   \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
        return map.at(quad).get();                                                                                      \
    }

    ELEMENT_METRICS(jacobians, elem->jacobian(xi, metric), J, 4)
    
    ELEMENT_METRICS(measures, *metric = elem->measure(xi), detJ, 1)

    ELEMENT_METRICS(physical_coordinates, elem->physical_coordinates(xi, metric), x, 2)

    #define EDGE_METRICS_ETYPE(name, fun, map, dim)                                                                         \
    const double * Mesh2D::edge_ ##name (const QuadratureRule * quad, EdgeType edge_type) const                             \
    {                                                                                                                       \
       auto& metric_map = (edge_type == INTERIOR) ? map## _int : map## _ext;                                                \
       if (not metric_map.contains(quad))                                                                                   \
       {                                                                                                                    \
            const int m = quad->n;                                                                                          \
            const int ne = n_edges(edge_type);                                                                              \
            double * ptr = metric_map.insert({quad, std::unique_ptr<double[]>(new double[dim*m*ne])}).first->second.get();  \
                                                                                                                            \
            const int sizes[] = {dim, m, ne};                                                                               \
            double xi;                                                                                                      \
                                                                                                                            \
            for (int e = 0; e < ne; ++e)                                                                                    \
            {                                                                                                               \
                const Edge * E = edge(e, edge_type);                                                                        \
                for (int i = 0; i < m; ++i)                                                                                 \
                {                                                                                                           \
                    xi = quad->x[i];                                                                                        \
                    double * metric = ptr + tensor_index(sizes, 0, i, e);                                                   \
                    fun;                                                                                                    \
                }                                                                                                           \
            }                                                                                                               \
       }                                                                                                                    \
       return metric_map.at(quad).get();                                                                                    \
    }

    #define EDGE_METRICS(name, fun, map, dim)                                                                               \
    const double * Mesh2D::edge_ ##name (const QuadratureRule * quad) const                                                 \
    {                                                                                                                       \
       if (not map.contains(quad))                                                                                          \
       {                                                                                                                    \
            const int m = quad->n;                                                                                          \
            const int ne = n_edges();                                                                                       \
            double * ptr = map.insert({quad, std::unique_ptr<double[]>(new double[dim*m*ne])}).first->second.get();         \
                                                                                                                            \
            const int sizes[] = {dim, m, ne};                                                                               \
            double xi;                                                                                                      \
                                                                                                                            \
            for (int e = 0; e < ne; ++e)                                                                                    \
            {                                                                                                               \
                const Edge * E = edge(e);                                                                                   \
                for (int i = 0; i < m; ++i)                                                                                 \
                {                                                                                                           \
                    xi = quad->x[i];                                                                                        \
                    double * metric = ptr + tensor_index(sizes, 0, i, e);                                                   \
                    fun;                                                                                                    \
                }                                                                                                           \
            }                                                                                                               \
       }                                                                                                                    \
       return map.at(quad).get();                                                                                           \
    }

    EDGE_METRICS(normals, E->normal(xi, metric), n, 2)
    EDGE_METRICS_ETYPE(normals, E->normal(xi, metric), n, 2)

    EDGE_METRICS(physical_coordinates, E->physical_coordinates(xi, metric), edge_x, 2)
    EDGE_METRICS_ETYPE(physical_coordinates, E->physical_coordinates(xi, metric), edge_x, 2)

    EDGE_METRICS(measures, *metric = E->measure(xi), edge_meas, 1)
    EDGE_METRICS_ETYPE(measures, *metric = E->measure(xi), edge_meas, 1)

} // namespace dg
