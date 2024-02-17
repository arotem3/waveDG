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

    void Face1D::serialize(util::Serializer& serializer) const
    {
        serializer.types.push_back(0);

        int start = serializer.offsets_int.back();
        serializer.offsets_int.push_back(start+4);
        serializer.data_int.push_back(id);
        serializer.data_int.push_back(elements[0]);
        serializer.data_int.push_back(elements[1]);
        serializer.data_int.push_back((int)type);

        start = serializer.offsets_double.back();
        serializer.offsets_double.push_back(start+0);
    }

    Face1D::Face1D(const int * data_ints, const double * data_doubles)
        : id{data_ints[0]}, elements{data_ints[1], data_ints[2]}, type{data_ints[3]} {}
#endif

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

    #ifdef WDG_USE_MPI
        double h0 = h;
        int success = MPI_Allreduce(&h0, &h, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh1D::min_h error: MPI_Allreduce failed.", success);
    #endif

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

    #ifdef WDG_USE_MPI
        double h0 = h;
        int success = MPI_Allreduce(&h0, &h, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (success != MPI_SUCCESS)
            wdg_error("Mesh1D::max_h error: MPI_Allreduce failed.", success);
    #endif

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

    Mesh1D Mesh1D::from_vertices(int nx, const double * x, bool periodic)
    {
        const int nel = nx - 1;

        Mesh1D mesh;
        mesh._elements.reserve(nel);

        // elements
        for (int el = 0; el < nel; ++el)
        {
            Element1D elem(x + el);
            elem.id = el;
            mesh._elements.push_back(std::move(elem));
        }

        // faces
        int face_id = 0;

        Face1D left_boundary;
        left_boundary.elements[1] = 0;
        left_boundary.id = face_id;
        if (periodic)
        {
            left_boundary.elements[0] = nel-1;
            left_boundary.type = FaceType::INTERIOR;
            mesh._interior_faces.push_back(face_id);
        }
        else
        {
            left_boundary.type = FaceType::BOUNDARY;
            mesh._boundary_faces.push_back(face_id);
        }
        mesh._faces.push_back(std::move(left_boundary));
        face_id++;

        for (; face_id < nel; ++face_id)
        {
            Face1D face;
            face.elements[0] = face_id-1;
            face.elements[1] = face_id;
            face.id = face_id;
            face.type = FaceType::INTERIOR;
            mesh._faces.push_back(std::move(face));
            mesh._interior_faces.push_back(face_id);
        }

        if (not periodic)
        {
            Face1D right_boundary;
            right_boundary.elements[0] = nel-1;
            right_boundary.id = face_id;
            right_boundary.type = FaceType::BOUNDARY;
            mesh._faces.push_back(std::move(right_boundary));
            mesh._boundary_faces.push_back(face_id);
        }

        return mesh;
    }

    Mesh1D Mesh1D::uniform_mesh(int nel, double a, double b, bool periodic)
    {
        dvec x(nel+1);
        for (int k = 0; k <= nel; ++k)
            x(k) = a + k * (b - a) / nel;

        return Mesh1D::from_vertices(nel+1, x, periodic);
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

                // determine which processors need which faces
                std::vector<std::set<int>> unique_faces(n_procs);
                for (auto& face : _faces)
                {
                    for (int s = 0; s < 2; ++s)
                    {
                        const int el = face.elements[s];
                        if (el < 0)
                            continue;
                        
                        const int proc = e2p.at(el);
                        unique_faces.at(proc).insert(face.id);
                    }
                }

                std::vector<util::Serializer> elements_to_send(n_procs-1);
                std::vector<Element1D> elements_root;

                std::vector<util::Serializer> faces_to_send(n_procs-1);
                std::vector<Face1D> faces_root;

                // serialize elements
                for (int el = 0; el < nel; ++el)
                {
                    const int p = e2p.at(el);

                    if (p == 0)
                        elements_root.push_back(std::move(_elements[el]));
                    else
                    {
                        auto& ser = elements_to_send.at(p-1);
                        _elements.at(el).serialize(ser);
                    }
                }

                // serialize faces
                for (int p = 1; p < n_procs; ++p)
                {
                    auto& ser = faces_to_send.at(p-1);
                    for (int face_id : unique_faces.at(p))
                    {
                        _faces.at(face_id).serialize(ser);
                    }
                }

                for (int face_id : unique_faces.at(0))
                {
                    faces_root.push_back(std::move(_faces.at(face_id)));
                }

                // send elements
                const int n_req = 2 * 6 * (n_procs-1);
                MPI_Request * reqs = new MPI_Request[n_req];

                MPI_Request * req = reqs;
                for (int p = 1; p < n_procs; ++p)
                {
                    auto& faces = faces_to_send.at(p-1);
                    auto& elems = elements_to_send.at(p-1);
                    
                    req = faces.send(req, MPI_COMM_WORLD, p);
                    req = elems.send(req, MPI_COMM_WORLD, p, 10);
                }

                // clear local data and move elements_root to elements vector
                _elements = std::move(elements_root);
                _faces = std::move(faces_root);
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
                util::Serializer faces_to_recv;
                MPI_Request reqs[10];

                auto req = faces_to_recv.recv(reqs, MPI_COMM_WORLD, 0);
                elems_to_recv.recv(req, MPI_COMM_WORLD, 0, 10);

                int success = MPI_Waitall(10, reqs, MPI_STATUSES_IGNORE);
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

                int nf = faces_to_recv.types.size();
                _faces.reserve(nf);
                for (int e = 0; e < nf; ++e)
                {
                    int t = faces_to_recv.types.at(e);
                    int start = faces_to_recv.offsets_int.at(e);
                    int *data_int = faces_to_recv.data_int.data() + start;
                    start = faces_to_recv.offsets_double.at(e);
                    double *data_double = faces_to_recv.data_double.data() + start;

                    switch (t)
                    {
                    case 0:
                        _faces.push_back(Face1D(data_int, data_double));
                        break;
                    default:
                        wdg_error("Mesh1D::distribute error: recieved face type not supported.");
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
            wdg_error("Mesh1D::distribute error: MPI_Bcase failed.", success);

        int k = 0;
        _elem_local_id.clear();
        for (auto& elem : _elements)
        {
            _elem_local_id[elem.id] = k;
            ++k;
        }

        k = 0;
        _face_local_id.clear();
        for (auto& face : _faces)
        {
            _face_local_id[face.id] = k;
            ++k;
        }
        
        _interior_faces.clear();
        _boundary_faces.clear();
        for (auto& face : _faces)
        {
            const int face_id = _face_local_id.at(face.id);
            if (face.type == FaceType::INTERIOR)
                _interior_faces.push_back(face_id);
            else
                _boundary_faces.push_back(face_id);
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
