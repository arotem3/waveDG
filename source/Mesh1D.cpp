#include "Mesh1D.hpp"

namespace dg
{
    // Element1D
    Element1D::Element1D(const double * x_) : ab{x_[0], x_[1]} {}

#ifdef WDG_USE_MPI
    void Element1D::serialize(util::Serializer& serializer) const
    {
        serializer.types.push_back(0);

        int start = serializer.offsets_int.back();
        serializer.offsets_int.push_back(start+1);
        serializer.data_int.push_back(id);

        start = serializer.offsets_double.back();
        serializer.offsets_double.push_back(start+2);
        serializer.data_double.push_back(ab[0]);
        serializer.data_double.push_back(ab[1]);
    }

    Element1D::Element1D(const int * data_ints, const double * data_doubles)
        : ab{data_doubles[0], data_doubles[1]}, id{data_ints[0]} {}
#endif

    double Element1D::physical_coordinates(double xi) const
    {
        return ab[0] + 0.5 * (ab[1] - ab[0]) * (xi + 1.0);
    }

    double Element1D::jacobian() const
    {
        return 0.5 * (ab[1] - ab[0]);
    }

    const double * Element1D::end_points() const
    {
        return ab;
    }

    Mesh1D::ElementMetricCollection::ElementMetricCollection(const Mesh1D& mesh_, const QuadratureRule* quad_)
        : mesh(mesh_), quad(quad_) {}

    template <typename EvalMetric>
    void set_element_metric(std::unique_ptr<double[]>& metric_, const Mesh1D& mesh, const QuadratureRule * quad, EvalMetric eval_metric)
    {
        const int m = quad->n;
        const int nel = mesh.n_elem();

        metric_.reset(new double[m * nel]);
        auto metric = reshape(metric_.get(), m, nel);

        for (int el = 0; el < nel; ++el)
        {
            auto& elem = mesh.element(el);
            for (int j = 0; j < m; ++j)
            {
                const double xi = quad->x[j];
                metric(j, el) = eval_metric(elem, xi);
            }
        }
    }

    const double * Mesh1D::ElementMetricCollection::jacobians() const
    {
        if (not J)
        {
            set_element_metric(J, mesh, quad, [](const Element1D& elem, double xi) -> double {return elem.jacobian();});
        }

        return J.get();
    }

    const double * Mesh1D::ElementMetricCollection::physical_coordinates() const
    {
        if (not x)
        {
            set_element_metric(x, mesh, quad, [](const Element1D& elem, double xi) -> double {return elem.physical_coordinates(xi);});
        }

        return x.get();
    }

    double Mesh1D::min_h() const
    {
        const int nel = n_elem();

        double h = std::numeric_limits<double>::infinity();
        for (int el = 0; el < nel; ++el)
        {
            h = std::min(h, element(el).jacobian());
        }

        return 2.0 * h;
    }

    double Mesh1D::max_h() const
    {
        const int nel = n_elem();

        double h = 0.0;
        for (int el = 0; el < nel; ++el)
        {
            h = std::max(h, element(el).jacobian());
        }

        return 2.0 * h;
    }

    const Mesh1D::ElementMetricCollection& Mesh1D::element_metrics(const QuadratureRule * quad) const
    {
        if (not elem_collections.contains(quad))
        {
            elem_collections.insert({quad, ElementMetricCollection(*this, quad)});
        }

        return elem_collections.at(quad);
    }

    Mesh1D Mesh1D::from_vertices(int nx, const double * x)
    {
        const int nel = nx - 1;

        Mesh1D mesh;
        mesh._elements.reserve(nel);

        for (int el = 0; el < nel; ++el)
        {
            Element1D elem(x + el);
            elem.id = el;
            mesh._elements.push_back(std::move(elem));
        }

        return mesh;
    }

    Mesh1D Mesh1D::uniform_mesh(int nel, double a, double b)
    {
        const double h = (b - a) / nel;

        Mesh1D mesh;
        mesh._elements.reserve(nel);

        for (int el = 0; el < nel; ++el)
        {
            const double x[2] = {a + h*el, a + h*(el+1)};
            Element1D elem(x);
            elem.id = el;
            mesh._elements.push_back(std::move(elem));
        }

        return mesh;
    }

#ifdef WDG_USE_MPI
    void Mesh1D::distribute()
    {
        int n_procs;
        MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

        const bool distributed = n_procs > 1;

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == 0)
        {
            const int nel = n_elem();

            e2p.resize(nel);
            for (int el = 0; el < nel; ++el)
            {
                e2p[el] = (el * n_procs) / nel;
            }
        }

        if (distributed)
        {
            if (rank == 0)
            {
                const int nel = n_elem();

                std::vector<util::Serializer> elements_to_send(n_procs-1);
                std::vector<Element1D> elements_root;

                for (int el = 0; el < nel; ++el)
                {
                    const int p = e2p.at(el);

                    if (p == 0)
                    {
                        elements_root.push_back(std::move(_elements[el]));
                    }
                    else
                    {
                        auto& ser = elements_to_send.at(p-1);
                        _elements.at(el).serialize(ser);
                    }
                }

                // send elements
                const int n_req = 6 * (n_procs-1);
                MPI_Request * reqs = new MPI_Request[n_req];

                MPI_Request * req = reqs;
                for (int p = 1; p < n_procs; ++p)
                {
                    auto& ser = elements_to_send.at(p-1);
                    
                    req = ser.send(req, MPI_COMM_WORLD, p);
                }

                // clear local data and move elements_root to elements vector
                _elements = std::move(elements_root);
                elem_collections.clear();

                int success = MPI_Waitall(n_req, reqs, MPI_STATUSES_IGNORE);
                if (success != MPI_SUCCESS)
                    wdg_error("Mesh1D::distribute error: MPI_Waitall failed.", success);
            }
            else
            {
                _elements.clear();
                elem_collections.clear();
                e2p.clear();

                util::Serializer elems_to_recv;
                MPI_Request reqs[5];

                elems_to_recv.recv(reqs, MPI_COMM_WORLD, 0);

                int success = MPI_Waitall(5, reqs, MPI_STATUSES_IGNORE);
                if (success != MPI_SUCCESS)
                    wdg_error("Mesh1D::distribute error: MPI_Waitall failed.", success);

                 int nel = elems_to_recv.types.size();
                _elements.reserve(nel);

                for (int el = 0; el < nel; ++el)
                {
                    int t = elems_to_recv.types.at(el);
                    int start = elems_to_recv.offsets_int.at(el);
                    int *data_int = elems_to_recv.data_int.data() + start;
                    start = elems_to_recv.offsets_double.at(el);
                    double *data_double = elems_to_recv.data_double.data() + start;

                    switch (t)
                    {
                    case 0: // QuadElement
                        _elements.push_back(Element1D(data_int, data_double));
                        break;
                    default:
                        wdg_error("Mesh1D::distribute error: recieved element type not supported.");
                        break;
                    }
                }
            }
        }
    
        int global_nel = global_n_elem();

        if (rank != 0)
        {
            e2p.resize(global_nel);
        }

        int success = MPI_Bcast(e2p.data(), global_nel, MPI_INT, 0, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh1D::distribute error: MPI_Bcase failed.", success);

        int k = 0;
        for (auto& elem : _elements)
        {
            _elem_local_id[elem.id] = k;
            ++k;
        }
    }

    int Mesh1D::global_n_elem() const
    {
        int global_nel;
        int local_nel = n_elem();
        int success = MPI_Allreduce(&local_nel, &global_nel, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh1D::global_n_elem error: MPI_Allreduce failed.", success);

        return global_nel;
    }
#endif
} // namespace dg
