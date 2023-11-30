#include "FaceProlongator.hpp"

namespace 
{
    typedef std::map<int, std::vector<int>> comm_patern;   
}

namespace dg
{
#ifdef WDG_USE_MPI
    LobattoFaceProlongator::LobattoFaceProlongator(int nv, const Mesh2D& mesh, const QuadratureRule * basis, Edge::EdgeType edge_type_)
        : n_elem(mesh.n_elem()),
          n_edges(mesh.n_edges(edge_type_)),
          n_colloc(basis->n),
          n_var(nv),
          edge_type(edge_type),
          v2e(n_colloc, 2, n_edges)
    {
        if (n_edges == 0)
            return;

        v2e.fill(-1);

        const int nc = n_colloc;
        auto mapV2E = [nc,&mesh](int i, int s, int el) -> int
        {
            el = mesh.local_element_index(el);
            const int m = (s == 0 || s == 2) ? i : (s == 1) ? (nc-1) : 0;
            const int n = (s == 1 || s == 3) ? i : (s == 2) ? (nc-1) : 0;

            return m + nc * (n + nc * el);
        };

        if (edge_type == Edge::BOUNDARY)
        {
            for (int e = 0; e < n_edges; ++e)
            {
                const Edge * edge = mesh.edge(e, edge_type);
                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];

                for (int i = 0; i < n_colloc; ++i)
                {
                    v2e(i, 0, e) = mapV2E(i, s0, el0);
                }
            }
        }
        else
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            std::map<int, std::vector<std::pair<int,int>>> sr;

            for (int e = 0; e < n_edges; ++e)
            {
                const Edge * edge = mesh.edge(e, edge_type);

                const int el0 = edge->elements[0];
                const int s0 = edge->sides[0];
                const int owner0 = mesh.find_element(el0);

                if (owner0 == rank)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        v2e(i, 0, e) = mapV2E(i, s0, el0);
                    }
                    local_face_pattern.push_back(0 + 2*e);
                }

                const int el1 = edge->elements[1];
                const int s1 = edge->sides[1];
                const int owner1 = mesh.find_element(el1);

                if (owner1 == rank)
                {
                    for (int i = 0; i < n_colloc; ++i)
                    {
                        const int j = (edge->delta > 0) ? i : (n_colloc - 1 - i);
                        v2e(i, 1, e) = mapV2E(j, s1, el1);
                    }
                    local_face_pattern.push_back(1 + 2*e);
                }

                if (owner0 != owner1)
                {
                    const int partner = (owner0 == rank) ? owner1 : owner0;
                    const int s = (owner0 == rank) ? 0 : 1;

                    sr[partner].push_back({s + 2*e, 1-s + 2*e}); // {send, recv}
                }
            }

            const int edge_size = n_var * n_colloc;
            
            const int n_channels = sr.size();
            rreq.init(n_channels);
            sreq.init(n_channels);
            channels.resize(n_channels);

            int k = 0;
            for (auto& [partner, I] : sr)
            {
                const int n_edges_sendrecv = I.size();
                const int msg_size = edge_size * n_edges_sendrecv;

                auto& channel = channels[k];

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
    }

    void LobattoFaceProlongator::action(const double * u_, double * uf_) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        uf.zeros();

        if (edge_type == Edge::BOUNDARY)
        {
            for (int e = 0; e < n_edges; ++e)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = v2e(i, 0, e);

                    for (int d = 0; d < n_var; ++d)
                    {    
                        uf(i, d, 0, e) = u(d, j);
                    }
                }
            }
        }
        else
        {
            // prolongate all face values from local elements
            const int n = local_face_pattern.size();
            for (int l = 0; l < n; ++l)
            {
                const int s = local_face_pattern[l] % 2;
                const int e = local_face_pattern[l] / 2;

                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = v2e(i, s, e);

                    for (int d = 0; d < n_var; ++d)
                    {    
                        uf(i, d, s, e) = u(d, j);
                    }
                }
            }

            // initialize recieves
            int status = MPI_Startall(rreq.size(), rreq.get());
            mpi_error_and_abort_on_fail("MPI_Startall", status);

            // copy to buffer and send
            for (auto& channel : channels)
            {
                const int partner = channel.partner;
                auto& l2p = channel.l2p;
                auto& sbuf = channel.send_buf;

                const int n_edges_to_send = l2p.size();

                int k = 0;
                for (int l = 0; l < n_edges_to_send; ++l)
                {
                    const int s = l2p(l) % 2;
                    const int e = l2p(l) / 2;

                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            sbuf(k) = uf(i, d, s, e);
                            ++k;
                        }
                    }
                }
            }

            status = MPI_Startall(sreq.size(), sreq.get());
            mpi_error_and_abort_on_fail("MPI_Startall", status);

            // wait for recv
            status = MPI_Waitall(rreq.size(), rreq.get(), MPI_STATUSES_IGNORE);
            mpi_error_and_abort_on_fail("MPI_Waitall", status);

            // copy recieved values to uf
            for (auto& channel : channels)
            {
                auto& p2l = channel.p2l;
                auto& rbuf = channel.recv_buf;

                const int n_edges_to_recv = p2l.size();
                
                int k = 0;
                for (int l = 0; l < n_edges_to_recv; ++l)
                {
                    const int s = p2l(l) % 2;
                    const int e = p2l(l) / 2;

                    for (int i = 0; i < n_colloc; ++i)
                    {
                        for (int d = 0; d < n_var; ++d)
                        {
                            uf(i, d, s, e) = rbuf(k);
                            ++k;
                        }
                    }
                }
            }

            // wait for send (just in case)
            status = MPI_Waitall(sreq.size(), sreq.get(), MPI_STATUSES_IGNORE);
            mpi_error_and_abort_on_fail("MPI_Waitall", status);
        }
    }

    void LobattoFaceProlongator::t(const double * uf_, double * u_) const
    {
        if (n_edges == 0)
            return;

        auto uf = reshape(uf_, n_colloc, n_var, 2, n_edges);
        auto u = reshape(u_, n_var, n_colloc * n_colloc * n_elem);

        if (edge_type == Edge::BOUNDARY)
        {
            for (int e = 0; e < n_edges; ++e)
            {
                for (int i = 0; i < n_colloc; ++i)
                {
                    const int j = v2e(i, 0, e);
                    
                    for (int d = 0; d < n_var; ++d)
                    {
                        u(d, j) += uf(i, d, 0, e);
                    }
                }
            }
        }
        else
        {
            const int n = local_face_pattern.size();
            for (int l = 0; l < n; ++l)
            {
                const int s = local_face_pattern[l] % 2;
                const int e = local_face_pattern[l] / 2;

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
    }

#else
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
#endif
} // namespace dg