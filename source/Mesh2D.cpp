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
                    edge->type = Edge::BOUNDARY;
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
                    edge->type = Edge::INTERIOR;
                    edge->delta = (C0 == n1) ? 1 : -1;
                }
            }
        }

        // boundary edges
        for (const auto& edge : mesh._edges)
        {
            if (edge->type == Edge::BOUNDARY)
                mesh._boundary_edges.push_back(edge->id);
            else
                mesh._interior_edges.push_back(edge->id);
        }

        return mesh;
    }

    Mesh2D Mesh2D::from_file(const std::string& dir)
    {
        std::ifstream info(dir + "/info.txt");
        if (not info)
            throw std::runtime_error("cannot open file: " + dir + "/info.txt");

        int n_pts, n_elem;
        info >> n_pts >> n_elem;
        info.close();

        dmat x(2, n_pts);
        Matrix<int> elems(4, n_elem);

        std::ifstream coo(dir + "/coordinates.txt");
        if (not coo)
            throw std::runtime_error("cannot open file: " + dir + "/coordinates.txt");

        for (int i = 0; i < n_pts; ++i)
        {
            coo >> x(0, i) >> x(1, i);
        }
        coo.close();

        std::ifstream elements(dir + "/elements.txt");
        if (not coo)
            throw std::runtime_error("cannot open file: " + dir + "/elements.txt");

        for (int i = 0; i < n_elem; ++i)
        {
            elements >> elems(0, i) >> elems(1, i) >> elems(2, i) >> elems(3, i);
        }
        elements.close();

        return Mesh2D::from_vertices(n_pts, x.data(), n_elem, elems.data());
    }

    Mesh2D Mesh2D::uniform_rect(int nx, double ax, double bx, int ny, double ay, double by)
    {
        int np = (nx+1)*(ny+1);
        int nel = nx*ny;
        dcube coo(2, nx+1, ny+1);
        Cube<int> elems(4, nx, ny);

        auto l = [nx,ny](int i, int j) -> int {return i + (nx+1)*j;};

        double dx = (bx - ax) / nx;
        double dy = (by - ay) / ny;
        for (int j=0; j <= ny; ++j)
        {
            const double y = ay + dy * j;
            for (int i=0; i <= nx; ++i)
            {
                coo(0, i, j) = ax + dx * i;
                coo(1, i, j) = y;
            }
        }
        
        for (int j=0; j < ny; ++j)
        {
            for (int i=0; i < nx; ++i)
            {
                elems(0, i, j) = l(  i,   j);
                elems(1, i, j) = l(i+1,   j);
                elems(2, i, j) = l(i+1, j+1);
                elems(3, i, j) = l(  i, j+1);
            }
        }

        return from_vertices(np, coo, nel, elems);
    }
    double Mesh2D::min_element_measure() const
    {
        double h = std::numeric_limits<double>::infinity();
        for (auto& elem : _elements)
        {
            h = std::min(h, elem->area());
        }
        return h;
    }

    double Mesh2D::max_element_measure() const
    {
        double h = -1;
        for (auto& elem : _elements)
        {
            h = std::max(h, elem->area());
        }
        return h;
    }

    double Mesh2D::min_edge_measure() const
    {
        double h = std::numeric_limits<double>::infinity();
        for (auto& edge : _edges)
        {
            h = std::min(h, edge->length());
        }
        return h;
    }

    double Mesh2D::max_edge_measure() const
    {
        double h = -1;
        for (auto& edge : _edges)
        {
            h = std::max(h, edge->length());
        }
        return h;
    }
    
    template <typename MetricEval>
    const double * Mesh2D::element_metric(int dim, MetricCollection& map, const QuadratureRule * quad, MetricEval fun) const
    {
        if (not map.contains(quad))
        {
            const int m = quad->n;
            const int nel = n_elem();

            double * ptr = map.insert({quad, std::unique_ptr<double[]>(new double[dim*m*m*nel])}).first->second.get();

            const int sizes[] = {dim, m, m, nel};
            double xi[2];

            for (int el = 0; el < nel; ++el)
            {
                const Element * elem = element(el);
                for (int j = 0; j < m; ++j)
                {
                    xi[1] = quad->x[j];
                    for (int i = 0; i < m; ++i)
                    {
                        xi[0] = quad->x[i];
                        double * metric = ptr + tensor_index(sizes, 0, i, j, el);

                        fun(metric, elem, xi);
                    }
                }
            }
        }
        return map.at(quad).get();
    }

    const double * Mesh2D::element_jacobians(const QuadratureRule * quad) const
    {
        return element_metric(4, J, quad, [](double* metric, const Element* elem, const double* xi)->void{elem->jacobian(xi, metric);});
    }

    const double * Mesh2D::element_measures(const QuadratureRule * quad) const
    {
        return element_metric(1, detJ, quad, [](double* metric, const Element* elem, const double* xi)->void{*metric = elem->measure(xi);});
    }

    const double * Mesh2D::element_physical_coordinates(const QuadratureRule * quad) const
    {
        return element_metric(2, x, quad, [](double* metric, const Element* elem, const double* xi)->void{elem->physical_coordinates(xi, metric);});
    }

    template <bool byEdgeType, typename MetricEval>
    const double * Mesh2D::edge_metric(int dim, Edge::EdgeType etype, MetricCollection& map, const QuadratureRule * quad, MetricEval fun) const
    {
        if (not map.contains(quad))
        {
            const int m = quad->n;
            int ne = (byEdgeType) ? n_edges(etype) : n_edges();

            double * ptr = map.insert({quad, std::unique_ptr<double[]>(new double[dim*m*ne])}).first->second.get();

            const int sizes[] = {dim, m, ne};
            double xi;

            for (int e = 0; e < ne; ++e)
            {
                const Edge * E = (byEdgeType) ? edge(e, etype) : edge(e);

                for (int i = 0; i < m; ++i)
                {
                    xi = quad->x[i];
                    double * metric = ptr + tensor_index(sizes, 0, i, e);

                    fun(metric, E, xi);
                }
            }
        }
        return map.at(quad).get();
    }

    const double * Mesh2D::edge_normals(const QuadratureRule * quad) const
    {
        Edge::EdgeType dummy = Edge::INTERIOR;
        return edge_metric<false>(2, dummy, n, quad, [](double* metric, const Edge * E, double xi)-> void {E->normal(xi, metric);});
    }

    const double *Mesh2D::edge_normals(const QuadratureRule * quad, Edge::EdgeType type) const
    {
        auto& nb = (type == Edge::INTERIOR) ? n_int : n_ext;
        return edge_metric<true>(2, type, nb, quad, [](double* metric, const Edge * E, double xi)-> void {E->normal(xi, metric);});
    }

    const double * Mesh2D::edge_physical_coordinates(const QuadratureRule * quad) const
    {
        Edge::EdgeType dummy = Edge::INTERIOR;
        return edge_metric<false>(2, dummy, edge_x, quad, [](double* metric, const Edge * E, double xi)-> void {E->physical_coordinates(xi, metric);});
    }

    const double *Mesh2D::edge_physical_coordinates(const QuadratureRule * quad, Edge::EdgeType type) const
    {
        auto& xb = (type == Edge::INTERIOR) ? edge_x_int : edge_x_ext;
        return edge_metric<true>(2, type, xb, quad, [](double* metric, const Edge * E, double xi)-> void {E->physical_coordinates(xi, metric);});
    }

    const double * Mesh2D::edge_measures(const QuadratureRule * quad) const
    {
        Edge::EdgeType dummy = Edge::INTERIOR;
        return edge_metric<false>(1, dummy, edge_meas, quad, [](double* metric, const Edge * E, double xi)-> void {*metric = E->measure(xi);});
    }

    const double *Mesh2D::edge_measures(const QuadratureRule * quad, Edge::EdgeType type) const
    {
        auto& mb = (type == Edge::INTERIOR) ? edge_meas_int : edge_meas_ext;
        return edge_metric<true>(1, type, mb, quad, [](double* metric, const Edge * E, double xi)-> void {*metric = E->measure(xi);});
    }
} // namespace dg
