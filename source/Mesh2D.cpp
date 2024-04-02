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
                    edge->type = FaceType::BOUNDARY;
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
                    edge->type = FaceType::INTERIOR;
                    edge->delta = (C0 == n1) ? 1 : -1;
                }
            }
        }

        // boundary edges
        for (const auto& edge : mesh._edges)
        {
            if (edge->type == FaceType::BOUNDARY)
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
            wdg_error("Mesh2D::from_file error: cannot open file: "  + dir + "/info.txt");

        int n_pts, n_elem;
        info >> n_pts >> n_elem;
        info.close();

        dmat x(2, n_pts);
        Matrix<int> elems(4, n_elem);

        std::ifstream coo(dir + "/coordinates.txt");
        if (not coo)
            wdg_error("Mesh2D::from_file error: cannot open file: "  + dir + "/coordinates.txt");

        for (int i = 0; i < n_pts; ++i)
        {
            coo >> x(0, i) >> x(1, i);
        }
        coo.close();

        std::ifstream elements(dir + "/elements.txt");
        if (not coo)
            wdg_error("Mesh2D::from_file error: cannot open file: "  + dir + "/elements.txt");

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

#ifdef WDG_USE_MPI
    struct _element_center
    {
        int id;
        double x[2];
        double pc;
    };

    static bool operator<(const _element_center& e1, const _element_center& e2)
    {
        return e1.pc < e2.pc;
    }

    // computes first principal component of the coordinates of the element
    // centers in the range [first, last) (not including last)
    template <typename Iter>
    static void pca(Iter first, Iter last)
    {
        double xm = 0.0;
        double ym = 0.0;
        double variance_x = 0.0;
        double variance_y = 0.0;
        double covariance = 0.0;

        const int n = std::distance(first, last);

        // compute mean
        for (Iter u=first; u != last; ++u)
        {
            xm += u->x[0];
            ym += u->x[0];
        }
        xm /= n;
        ym /= n;

        // compute covariance
        for (Iter u=first; u != last; ++u)
        {
            const double s = (u->x[0] - xm);
            const double t = (u->x[1] - ym);
            variance_x += s*s / (n-1);
            variance_y += t*t / (n-1);
            covariance += s*t / (n-1);
        }

        const double determinant = std::sqrt(std::pow(variance_x - variance_y, 2) + std::pow(2*covariance, 2));
        const double eigenvec_x = (determinant + variance_x - variance_y) / (2*covariance);
        const double eigenvec_y = 1.0;

        for (Iter u=first; u != last; ++u)
        {
            u->pc = eigenvec_x * (u->x[0] - xm) + eigenvec_y * (u->x[1] - ym);
        }
    }

    static int smallest_prime_divisor(int n)
    {
        if (n % 2 == 0)
            return 2;
        
        for (int k=3; k*k <= n; k += 2)
        {
            if (n % k == 0)
                return k;
        }

        return n;
    }

    // recursive coordinate bisection
    static std::vector<int> rcb(const std::vector<std::unique_ptr<Element>>& elements, int num_procs)
    {
        const int nel = elements.size();

        std::vector<_element_center> a(nel);
        std::vector<int> e2p(nel);

        // compute mesh split
        constexpr double zero[] = {0.0, 0.0};

        for (int el = 0; el < nel; ++el)
        {
            auto &u = a.at(el);
            elements.at(el)->physical_coordinates(zero, u.x);
            u.id = el;

            e2p.at(el) = 0;
        }

        if (num_procs < 2)
            return e2p;
            
        std::queue<std::tuple<int, int, int>> q;
        q.push(std::make_tuple(0, nel, num_procs));

        int rank = 1;
        while (not q.empty())
        {
            const auto [first, last, p] = q.front(); q.pop();

            pca(a.data() + first, a.data() + last);

            const int divisor = smallest_prime_divisor(p); // number of partitions to make
            const int next_p = p / divisor;

            Vec<int> pos(divisor+1);
            for (int k = 0; k <= divisor; ++k)
                pos(k) = first + k * (last - first) / divisor;

            // sort
            for (int k = 1; k < divisor; ++k)
                std::nth_element(a.data() + pos(k-1), a.data() + pos(k), a.data() + last);

            // set rank
            for (int k = 1; k < divisor; ++k, ++rank)
            {
                for (int i = pos(k); i < pos(k+1); ++i)
                {
                    e2p.at(a.at(i).id) = rank;
                }
            }

            // next split
            if (next_p > 1)
            {
                for (int k = 0; k < divisor; ++k)
                {
                    q.push(std::make_tuple(pos(k), pos(k+1), next_p));
                }
            }
        }

        return e2p;
    }

    // node in graph minimum degree
    static int min_degree(const std::vector<std::vector<int>>& a)
    {
        const int n = a.size();
        int k_min = 0, min_degree = std::numeric_limits<int>::max();
        for (int k=0; k < n; ++k)
        {
            if ((int)a.at(k).size() < min_degree)
            {
                k_min = k;
                min_degree = a.at(k).size();
            }
        }

        return k_min;
    }

    // Reverse Cuthill-McKee on element connectivity matrix where connectivity
    // is determined by shared edges. (shared nodes do not matter).
    static std::vector<int> rcm(int n_elem, const std::vector<std::unique_ptr<Edge>>& edges, int num_procs)
    {
        // construct adjacency graph
        std::vector<std::vector<int>> a(n_elem);

        for (auto& edge : edges)
        {
            if (edge->type == FaceType::INTERIOR)
            {
                const int e0 = edge->elements[0];
                const int e1 = edge->elements[1];

                a.at(e0).push_back(e1);
                a.at(e1).push_back(e0);
            }
        }

        // rcm
        std::queue<int> q;
        std::unordered_set<int> elements; // elements already in permutation
        std::vector<int> p; // permutation -> new ordering of elements
        p.reserve(n_elem);

        q.push(min_degree(a));

        while (not q.empty())
        {
            const int k = q.front(); q.pop();

            if (not elements.contains(k))
            {
                elements.insert(k);
                p.push_back(k);

                auto v = a.at(k); // copy v, at most v.size() = 4.
                
                // sort v by degree
                std::sort(v.begin(), v.end(), [&a](int i, int j) -> bool {return a.at(i).size() < a.at(j).size();});
                
                for (int vk : v)
                    q.push(vk);
            }
        }

        // verify that full permutation was computed
        if (not ((int)p.size() == n_elem))
            wdg_error("Mesh2D::distribute error: Reverse Cuthill-McKee failed to computed correct reordering of mesh elements. This may be a result of a mesh which has disjoint parts.");

        std::vector<int> e2p(n_elem);
        auto it = p.crbegin(); // read permutation in reverse order
        for (int k = 0; k < num_procs; ++k)
        {
            int start = k * n_elem / num_procs;
            int end = (k + 1) * n_elem / num_procs;

            for (int i = start; i < end; ++i)
            {
                e2p.at(*it) = k;
                ++it;
            }
        }

        return e2p;
    }

    void Mesh2D::distribute(const std::string& alg)
    {
        int num_procs;
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        const bool distributed = num_procs > 1;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // determine mesh splitting and construct e2p
        if (rank == 0)
        {
            const int nel = n_elem();

            if (distributed)
            {
                if (alg == "rcm")
                    e2p = rcm(nel, _edges, num_procs);
                else if (alg == "rcb")
                    e2p = rcb(_elements, num_procs);
                else
                    wdg_error("Mesh2D::scatter does not support algorithm: \"" + alg + "\". Algorithm must be one of {\"rcm\" (reverse Cuthill-mcKee), \"rcb\" (recursive coordinate bisection)}.");
            }
            else
            {
                e2p.resize(nel, 0); // all elements on root
            }
        }

        // distribute
        if (distributed)
        {
            if (rank == 0)
            {
                const int nel = n_elem();
                
                std::vector<std::set<int>> unique_edges(num_procs);
                for (auto& edge : _edges)
                {
                    const int e0 = edge->elements[0];
                    const int p0 = e2p.at(e0);
                    
                    unique_edges.at(p0).insert(edge->id);

                    if (edge->type == FaceType::INTERIOR)
                    {
                        const int e1 = edge->elements[1];
                        const int p1 = e2p.at(e1);

                        unique_edges.at(p1).insert(edge->id);
                    }
                }

                // distribute mesh
                std::vector<util::Serializer> edges_to_send(num_procs-1);
                std::vector<util::Serializer> elements_to_send(num_procs-1);

                std::vector<std::unique_ptr<Edge>> edges_root;
                std::vector<std::unique_ptr<Element>> elements_root;

                for (int el = 0; el < nel; ++el)
                {
                    int p = e2p.at(el);

                    if (p == 0)
                        elements_root.push_back(std::move(_elements[el]));
                    else
                    {
                        auto& ser = elements_to_send.at(p-1);
                        _elements.at(el)->serialize(ser);
                    }
                }

                for (int p = 1; p < num_procs; ++p)
                {
                    auto& ser = edges_to_send.at(p-1);
                    for (int edge_id : unique_edges.at(p))
                    {
                        _edges.at(edge_id)->serialize(ser);
                    }
                }

                for (int edge_id : unique_edges.at(0))
                {
                    edges_root.push_back(std::move(_edges[edge_id]));
                }

                const int nreq = 2 * 6 * (num_procs-1);
                MPI_Request * sreq = new MPI_Request[nreq];

                MPI_Request * req = sreq;
                for (int p = 1; p < num_procs; ++p)
                {
                    auto& edges = edges_to_send.at(p-1);
                    auto& elems = elements_to_send.at(p-1);

                    req = edges.send(req, MPI_COMM_WORLD, p);
                    req = elems.send(req, MPI_COMM_WORLD, p, 10);
                }

                // while sending... clean up mesh
                reset();
                _elements = std::move(elements_root);
                _edges = std::move(edges_root);
                
                int success = MPI_Waitall(nreq, sreq, MPI_STATUSES_IGNORE);
                if (success != MPI_SUCCESS)
                    wdg_error("Mesh2D::distribute error: MPI_Waitall failed.", success);
            }
            else
            {
                reset();

                util::Serializer edges_to_recv;
                util::Serializer elems_to_recv;

                MPI_Request rreq[10];
                MPI_Request * req = edges_to_recv.recv(rreq, MPI_COMM_WORLD, 0);
                elems_to_recv.recv(req, MPI_COMM_WORLD, 0, 10);

                int success = MPI_Waitall(10, rreq, MPI_STATUSES_IGNORE);
                if (success != MPI_SUCCESS)
                    wdg_error("Mesh2D::distribute error: MPI_Waitall failed.", success);

                // elems
                int nel = elems_to_recv.types.size();
                _elements.resize(nel);
                for (int i=0; i < nel; ++i)
                {
                    int t = elems_to_recv.types.at(i);
                    int start = elems_to_recv.offsets_int.at(i);
                    int *data_int = elems_to_recv.data_int.data() + start;
                    start = elems_to_recv.offsets_double.at(i);
                    double *data_double = elems_to_recv.data_double.data() + start;

                    switch (t)
                    {
                    case 0: // QuadElement
                        _elements.at(i).reset(new QuadElement(data_int, data_double));
                        break;
                    default:
                        wdg_error("Mesh2D::distribute error: recieved element type not supported.");
                        break;
                    }
                }

                // edges
                int ne = edges_to_recv.types.size();
                _edges.resize(ne);
                for (int i=0; i < ne; ++i)
                {
                    int t = edges_to_recv.types.at(i);
                    int start = edges_to_recv.offsets_int.at(i);
                    int *data_int = edges_to_recv.data_int.data() + start;
                    start = edges_to_recv.offsets_double.at(i);
                    double *data_double = edges_to_recv.data_double.data() + start;

                    switch (t)
                    {
                    case 0:
                        _edges.at(i).reset(new StraightEdge(data_int, data_double));
                        break;
                    default:
                        wdg_error("Mesh2D::distribute error: recieved edge type not supported.");
                        break;
                    }
                }
            }
        }
        
        int global_nel = global_n_elem();

        if (rank != 0)
            e2p.resize(global_nel);
        
        int success = MPI_Bcast(e2p.data(), global_nel, MPI_INT, 0, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh2D::distribute error: MPI_Bcase failed.", success);

        // compute local indices
        int k = 0;
        for (auto& elem : _elements)
        {
            _elem_local_id[elem->id] = k;
            ++k;
        }

        k = 0;
        for (auto& edge : _edges)
        {
            _edge_local_id[edge->id] = k;
            ++k;
        }

        // compute edge types
        _interior_edges.clear(); _boundary_edges.clear();
        for (auto& edge : _edges)
        {
            int edge_id = local_edge_index(edge->id);
            if (edge->type == FaceType::INTERIOR)
                _interior_edges.push_back(edge_id);
            else
                _boundary_edges.push_back(edge_id);
        }

        // compute face pattern
        compute_face_pattern();
    }

    int Mesh2D::global_n_elem() const
    {
        int global_nel;
        int local_nel = n_elem();
        int success = MPI_Allreduce(&local_nel, &global_nel, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh2D::global_n_elem error: MPI_Allreduce failed.", success);

        return global_nel;
    }

    void Mesh2D::compute_face_pattern() const
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::vector<int> lfp;
        // interior
        int nf = n_edges(FaceType::INTERIOR);
        for (int e = 0; e < nf; ++e)
        {
            const Edge * E = edge(e, FaceType::INTERIOR);

            const int el0 = E->elements[0];
            const int owner0 = find_element(el0);

            if (owner0 == rank)
                lfp.push_back(0 + 2*e);

            const int el1 = E->elements[1];
            const int owner1 = find_element(el1);

            if (owner1 == rank)
                lfp.push_back(1 + 2*e);
        }

        int n = lfp.size();
        interior_face_pattern.reshape(n);
        for (int i=0; i < n; ++i)
            interior_face_pattern(i) = lfp.at(i);
        
        // boundary
        nf = n_edges(FaceType::BOUNDARY);
        boundary_face_pattern.reshape(nf);
        for (int e = 0; e < nf; ++e)
            boundary_face_pattern(e) = 2*e;
    }
#endif

    template <typename EvalMetric>
    static void set_element_metric(std::unique_ptr<double[]>& metric_, int dim, const Mesh2D& mesh, const QuadratureRule * quad, EvalMetric eval_metric)
    {
        const int m = quad->n;
        const int nel = mesh.n_elem();

        metric_.reset(new double[dim * m * m * nel]);
        auto metric = reshape(metric_.get(), dim, m, m, nel);

        double xi[2];

        for (int el=0; el < nel; ++el)
        {
            const Element * elem = mesh.element(el);
            for (int j = 0; j < m; ++j)
            {
                xi[1] = quad->x[j];
                for (int i = 0; i < m; ++i)
                {
                    xi[0] = quad->x[i];
                    eval_metric(&metric(0, i, j, el), elem, xi);
                }
            }
        }
    }

    const double * Mesh2D::ElementMetricCollection::jacobians() const
    {
        if (not J)
        {
            set_element_metric(J, 4, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {elem->jacobian(xi, metric);});
        }

        return J.get();
    }

    const double * Mesh2D::ElementMetricCollection::measures() const
    {
        if (not detJ)
        {
            set_element_metric(detJ, 1, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {*metric = elem->measure(xi);});
        }

        return detJ.get();
    }

    const double * Mesh2D::ElementMetricCollection::physical_coordinates() const
    {
        if (not x)
        {
            set_element_metric(x, 2, mesh, quad, [](double* metric, const Element * elem, const double * xi) -> void {elem->physical_coordinates(xi, metric);});
        }

        return x.get();
    }

    template <typename EvalMetric>
    static void set_edge_metric(std::unique_ptr<double[]>& metric_, int dim, FaceType edge_type, const Mesh2D& mesh, const QuadratureRule * quad, EvalMetric eval_metric)
    {
        const int m = quad->n;
        const int ne = mesh.n_edges(edge_type);

        metric_.reset(new double[dim * m * ne]);
        auto metric = reshape(metric_.get(), dim, m, ne);

        for (int e = 0; e < ne; ++e)
        {
            const Edge * E = mesh.edge(e, edge_type);

            for (int i = 0; i < m; ++i)
            {
                double xi = quad->x[i];
                
                eval_metric(&metric(0, i, e), E, xi);
            }
        }
    }

    const double * Mesh2D::EdgeMetricCollection::measures() const
    {
        if (not detJ)
        {
            set_edge_metric(detJ, 1, edge_type, mesh, quad, [](double* metric, const Edge * E, double xi) -> void {*metric = E->measure(xi);});
        }

        return detJ.get();
    }

    const double * Mesh2D::EdgeMetricCollection::physical_coordinates() const
    {
        if (not x)
        {
            set_edge_metric(x, 2, edge_type, mesh, quad, [](double * metric, const Edge * E, double xi) -> void {E->physical_coordinates(xi, metric);});
        }

        return x.get();
    }

    const double * Mesh2D::EdgeMetricCollection::normals() const
    {
        if (not n)
        {
            set_edge_metric(n, 2, edge_type, mesh, quad, [](double * metric, const Edge * E, double xi) -> void {E->normal(xi, metric);});
        }

        return n.get();
    }

    double Mesh2D::min_h() const
    {
        double h = std::numeric_limits<double>::infinity();
        for (auto& edge : _edges)
        {
            h = std::min(h, edge->length());
        }

    #ifdef WDG_USE_MPI
        double h0 = h;
        int success = MPI_Allreduce(&h0,&h,1,MPI_DOUBLE,MPI_MIN,MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh2D::min_h error: MPI_Allreduce failed.", success);
    #endif

        return h;
    }

    double Mesh2D::max_h() const
    {
        double h = -1;
        for (auto& edge : _edges)
        {
            h = std::max(h, edge->length());
        }

    #ifdef WDG_USE_MPI
        double h0 = h;
        int success = MPI_Allreduce(&h0,&h,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh2D::max_h error: MPI_Allreduce failed.", success);
    #endif

        return h;
    }

    const Mesh2D::ElementMetricCollection& Mesh2D::element_metrics(const QuadratureRule * quad) const
    {
        if (not elem_collections.contains(quad))
        {
            elem_collections.insert({quad, ElementMetricCollection(*this, quad)});
        }

        return elem_collections.at(quad);
    }

    const Mesh2D::EdgeMetricCollection& Mesh2D::edge_metrics(const QuadratureRule * quad, FaceType edge_type) const
    {
        auto& collection = (edge_type == FaceType::INTERIOR) ? interior_edge_collections : boundary_edge_collections;

        if (not collection.contains(quad))
        {
            collection.insert({quad, EdgeMetricCollection(*this, quad, edge_type)});
        }

        return collection.at(quad);
    }

} // namespace dg
