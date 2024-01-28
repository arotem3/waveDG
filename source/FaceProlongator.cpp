#include "FaceProlongator.hpp"

namespace dg
{
#ifdef WDG_USE_MPI
    FaceProlongator::FaceProlongator(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(2),
          n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          n_var(n_var_),
          edge_type(edge_type_)
    {
        if (edge_type == FaceType::INTERIOR)
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            std::map<int, std::vector<std::pair<int,int>>> sr; // indices of edges and sides to send and recv
            std::vector<int> lfp;

            for (int e = 0; e < n_edges; ++e)
            {
                const Edge * edge = mesh.edge(e, edge_type);

                const int el0 = edge->elements[0];
                const int owner0 = mesh.find_element(el0);

                if (owner0 == rank)
                {
                    lfp.push_back(0 + 2*e);
                }

                const int el1 = edge->elements[1];
                const int owner1 = mesh.find_element(el1);

                if (owner1 == rank)
                {
                    lfp.push_back(1 + 2*e);
                }

                if (owner0 != owner1)
                {
                    const int partner = (owner0 == rank) ? owner1 : owner0;
                    const int s = (owner0 == rank) ? 0 : 1;

                    sr[partner].push_back({s + 2*e, 1-s + 2*e}); // {send, recv}
                }
            }

            // copy lfp to local_face_pattern
            const int n = lfp.size();
            local_face_pattern.reshape(n);
            for (int i=0; i < n; ++i)
            {
                local_face_pattern(i) = lfp[i];
            }

            const int edge_size = n_var * n_colloc;
            
            const int n_channels = sr.size(); // number of processors to communicate with
            rreq.init(n_channels);
            sreq.init(n_channels);
            channels.resize(n_channels);

            // prepare persistant communicators
            int k = 0;
            for (auto& [partner, I] : sr)
            {
                const int n_edges_sendrecv = I.size();
                const int msg_size = edge_size * n_edges_sendrecv;

                auto& channel = channels.at(k);

                channel.partner = partner;
                channel.l2p.reshape(n_edges_sendrecv);
                channel.p2l.reshape(n_edges_sendrecv);
                channel.send_buf.reshape(msg_size);
                channel.recv_buf.reshape(msg_size);

                for (int i=0; i < n_edges_sendrecv; ++i)
                {
                    channel.l2p(i) = I[i].first;
                    channel.p2l(i) = I[i].second;
                }

                MPI_Recv_init(channel.recv_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, rreq.get()+k);
                MPI_Send_init(channel.send_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, sreq.get()+k);

                ++k;
            }
        }
        else
        {
            local_face_pattern.reshape(n_edges);
            for (int e = 0; e < n_edges; ++e)
            {
                local_face_pattern(e) = 2*e;
            }
        }
    }

    FaceProlongator::FaceProlongator(int n_var_, const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(1),
          n_elem(mesh.n_elem()),
          n_edges(),
          n_colloc(basis->n),
          n_var(n_var_),
          edge_type(edge_type_)
    {
        if (edge_type == FaceType::INTERIOR)
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            const int global_n_elem = mesh.global_n_elem();
            const int boundary_elements[] = {0, global_n_elem - 1};

            // left most edge
            if (mesh.element(0).id != boundary_elements[0])
            {
                PersistantChannel channel;
                channel.partner = mesh.find_element(mesh.element(0).id - 1);
                
                channel.l2p.reshape(1);
                channel.l2p(0) = mesh.element(0).id;

                channel.send_buf.reshape(n_var);

                channel.p2l.reshape(1);
                channel.p2l(0) = mesh.element(0).id-1;

                channel.recv_buf.reshape(n_var);

                channels.push_back(std::move(channel));
            }

            // right most edge
            if (mesh.element(n_elem-1).id != boundary_elements[1])
            {
                PersistantChannel channel;
                channel.partner = mesh.find_element(mesh.element(n_elem-1).id + 1);

                channel.l2p.reshape(1);
                channel.l2p(0) = mesh.element(0).id;

                channel.send_buf.reshape(n_var);

                channel.p2l.reshape(1);
                channel.p2l(0) = mesh.element(0).id+1;

                channel.recv_buf.reshape(n_var);

                channels.push_back(std::move(channel));
            }

            const int n_channels = channels.size();
            rreq.init(n_channels);
            sreq.init(n_channels);

            for (int k = 0; k < n_channels; ++k)
            {
                auto& channel = channels.at(k);

                MPI_Send_init(channel.send_buf.data(), n_var, MPI_DOUBLE, channel.partner, 0, MPI_COMM_WORLD, sreq.get()+k);
                MPI_Recv_init(channel.recv_buf.data(), n_var, MPI_DOUBLE, channel.partner, 0, MPI_COMM_WORLD, rreq.get()+k);
            }
        }
    }

    void FaceProlongator::sendrecv(TensorWrapper<4,double>& uf) const
    {
        if (dim == 1)
            wdg_error("FaceProlongator not yet implemented with MPI.");

        // initialize recieves
        int status = MPI_Startall(rreq.size(), rreq.get());
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Startall failed.", status);

        // copy to buffer and send
        for (auto& channel : channels)
        {
            auto& l2p = channel.l2p;
            const int n_edges_to_send = l2p.size();
            auto sbuf = reshape(channel.send_buf, n_var, n_colloc, n_edges_to_send);

            for (int l = 0; l < n_edges_to_send; ++l)
            {
                const int s = l2p(l) % 2;
                const int e = l2p(l) / 2;

                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        sbuf(d, i, l) = uf(i, d, s, e);
                    }
                }
            }
        }

        status = MPI_Startall(sreq.size(), sreq.get());
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Startall failed.", status);

        // wait for recv
        status = MPI_Waitall(rreq.size(), rreq.get(), MPI_STATUSES_IGNORE);
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Waitall failed.", status);

        // copy recieved values to uf
        for (auto& channel : channels)
        {
            auto& p2l = channel.p2l;
            const int n_edges_to_recv = p2l.size();
            auto rbuf = reshape(channel.recv_buf, n_var, n_colloc, n_edges_to_recv);
            
            for (int l = 0; l < n_edges_to_recv; ++l)
            {
                const int s = p2l(l) % 2;
                const int e = p2l(l) / 2;

                for (int i = 0; i < n_colloc; ++i)
                {
                    for (int d = 0; d < n_var; ++d)
                    {
                        uf(i, d, s, e) = rbuf(d, i, l);
                    }
                }
            }
        }

        // wait for send (just in case)
        status = MPI_Waitall(sreq.size(), sreq.get(), MPI_STATUSES_IGNORE);
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Waitall failed.", status);
    }

    LobattoFaceProlongator::LobattoFaceProlongator(int nv, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(nv, mesh, basis, edge_type_),
          v2e(n_colloc, 2, n_edges)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc,&mesh](int i, int f, int el) -> int
        {
            el = mesh.local_element_index(el);
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto const edge = mesh.edge(e, edge_type);

            const int el = edge->elements[s];
            const int f = edge->sides[s];
            const bool reversed = (s == 1) && (edge->delta < 0); // whether the degrees of freedom of the second element need to be reversed to match the first element.

            int j = (reversed) ? (n_colloc - 1) : 0;
            for (int i=0; i < n_colloc; ++i)
            {
                v2e(i, s, e) = mapV2E(j, f, el);
                j += (reversed) ? -1 : 1;
            }
        }
    }

    LobattoFaceProlongator::LobattoFaceProlongator(int nv, const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(nv, mesh, basis, edge_type_),
          v2e(2, n_edges, 1)
    {
        wdg_error("LobattoFaceProlongator not yet implemented using MPI.");
    }

    void LobattoFaceProlongator::action(const double * u_, double * uf_) const
    {
        if (dim == 1)
            wdg_error("LobattoFaceProlongator::action not yet implemented with MPI.");

        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        uf.zeros();

        // prolongate all face values from local elements
        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                const int j = v2e(i, s, e);

                for (int d = 0; d < n_var; ++d)
                {
                    uf(i, d, s, e) = u(d, j);
                }
            }
        }

        if (edge_type == FaceType::INTERIOR)
            sendrecv(uf);
    }

    void LobattoFaceProlongator::t(const double * uf_, double * u_) const
    {
        if (dim == 1)
            wdg_error("LobattoFaceProlongator::t not yet implemented with MPI.");
        
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                const int j = v2e(i, s, e);

                for (int d = 0; d < n_var; ++d)
                {
                    u(d, j) += uf(i, d, s, e);
                }
            }
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(n_var_, mesh, basis, edge_type_),
          v2e(n_colloc, n_colloc, 2, n_edges),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc](int k, int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1-k) : k;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1-k) : k;

            return m + nc * (n + nc * el);
        };

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            auto const edge = mesh.edge(e, edge_type);

            const int el = edge->elements[s];
            const int f = edge->sides[s];
            const bool reversed = (s == 1) && (edge->delta < 0); // whether the degrees of freedom of the second element need to be reversed to match the first element.

            int j = (reversed) ? (n_colloc - 1) : 0;
            for (int i=0; i < n_colloc; ++i)
            {
                for (int k=0; k < n_colloc; ++k)
                {
                    v2e(k, i, s, e) = mapV2E(k, j, f, el);
                }
                j += (reversed) ? -1 : 1;
            }
        }
    }

    LegendreFaceProlongator::LegendreFaceProlongator(int n_var, const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type)
        : FaceProlongator(n_var, mesh, basis, edge_type)
    {
        wdg_error("LegendreFaceProlongator not yet implemented with MPI.");
    }

    void LegendreFaceProlongator::action(const double * u_, double * uf_) const
    {
        if (dim == 1)
            wdg_error("LobattoFaceProlongator::action not yet implemented with MPI.");

        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        uf.zeros();

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double val = 0.0;
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, i, s, e);
                        val += u(d, j) * P(k);
                    }
                    uf(i, d, s, e) = val;
                }
            }
        }

        if (edge_type == FaceType::INTERIOR)
            sendrecv(uf);
    }

    void LegendreFaceProlongator::t(const double * uf_, double * u_) const
    {
        if (dim == 1)
            wdg_error("LobattoFaceProlongator::t not yet implemented with MPI.");
        
        if (n_edges == 0)
            return;
            
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        for (int l : local_face_pattern)
        {
            const int s = l % 2;
            const int e = l / 2;

            for (int i = 0; i < n_colloc; ++i)
            {
                for (int d = 0; d < n_var; ++d)
                {
                    double Y = uf(i, d, s, e);
                    for (int k = 0; k < n_colloc; ++k)
                    {
                        const int j = v2e(k, i, s, e);
                        u(d, j) += Y * P(k);
                    }
                }
            }
        }
    }
#else
    FaceProlongator::FaceProlongator(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(2),
          n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          n_var(n_var_),
          edge_type(edge_type_) {}

    FaceProlongator::FaceProlongator(int n_var_, const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : dim(1),
          n_elem(mesh.n_elem()),
          n_edges(mesh.n_faces(edge_type_)),
          n_colloc(basis->n),
          n_var(n_var_),
          edge_type(edge_type_) {}

    LobattoFaceProlongator::LobattoFaceProlongator(int nvar, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(nvar, mesh, basis, edge_type_),
          v2e(n_colloc, 2, n_edges)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc](int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1) : 0;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1) : 0;

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

            if (edge_type == FaceType::INTERIOR)
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

    LobattoFaceProlongator::LobattoFaceProlongator(int nvar, const Mesh1D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(nvar, mesh, basis, edge_type_), v2e(2, n_edges, 1)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);
        
        for (int e = 0; e < n_edges; ++e)
        {
            auto& face = mesh.face(e, edge_type);
            const int el0 = face.elements[0];
            v2e(0, e, 0) = (n_colloc-1) + n_colloc * el0;

            if (edge_type == FaceType::INTERIOR)
            {
                const int el1 = face.elements[1];
                v2e(1, e, 0) = (0) + n_colloc * el1;
            }
        }
    }

    void LobattoFaceProlongator::action(const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == FaceType::INTERIOR) ? 2 : 1;

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

    void LobattoFaceProlongator::t(const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == FaceType::INTERIOR) ? 2 : 1;

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

    LegendreFaceProlongator::LegendreFaceProlongator(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis, FaceType edge_type_)
        : FaceProlongator(n_var_, mesh, basis, edge_type_),
          v2e(n_colloc, n_colloc, 2, n_edges),
          P(n_colloc)
    {
        if (n_edges == 0)
            return;

        const double x = -1.0;
        lagrange_basis(P, n_colloc, basis->x, 1, &x);

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc](int k, int i, int f, int el) -> int
        {
            const int m = (f == 0 || f == 2) ? i : (f == 1) ? (nc-1-k) : k;
            const int n = (f == 1 || f == 3) ? i : (f == 2) ? (nc-1-k) : k;

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

            if (edge_type == FaceType::INTERIOR)
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

    void LegendreFaceProlongator::action(const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == FaceType::INTERIOR) ? 2 : 1;

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

    void LegendreFaceProlongator::t(const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;
            
        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        const int n_sides = (edge_type == FaceType::INTERIOR) ? 2 : 1;

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
#endif
} // namespace dg