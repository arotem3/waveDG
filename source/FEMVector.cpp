#include "FEMVector.hpp"

namespace dg
{
    FaceVector::FaceVector(int n_var_, const Mesh1D& mesh, FaceType face_type_, const QuadratureRule * basis)
        : dim{1},
          _n_var{n_var_},
          _n_basis{basis->n},
          _n_faces{mesh.n_faces(face_type_)},
          _face_type{face_type_},
          x(_n_basis, _n_var, 2, _n_faces)
    {
    #ifdef WDG_USE_MPI
        if (_face_type == FaceType::INTERIOR)
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            // determine send and recv pattern
            std::map<int, std::vector<std::pair<int,int>>> sr;

            for (int f = 0; f < _n_faces; ++f)
            {
                auto& face = mesh.face(f, _face_type);
                const int el0 = face.elements[0];
                const int el1 = face.elements[1];

                const int proc0 = mesh.find_element(el0);
                const int proc1 = mesh.find_element(el1);
                
                if ((proc0 != rank) && (proc1 != rank))
                    wdg_error("FaceProlongator error: Mesh incorrectly distributed. Mesh has face where both elements are not on this processor.");

                if (proc0 != proc1)
                {
                    const int partner = (proc0 == rank) ? proc1 : proc0;
                    const int s = (proc0 == rank) ? 0 : 1;

                    sr[partner].push_back({s + 2*f, 1-s + 2*f}); // {send, recv}
                }
            }

            const int n_channels = sr.size();
            channels.resize(n_channels);
            rreq.init(n_channels);
            sreq.init(n_channels);

            int k = 0;
            for (auto& [partner, I] : sr)
            {
                const int n_faces_sendrecv = I.size(); // 1, or 2
            #ifdef WDG_DEBUG
                if (n_faces_sendrecv > 2)
                    wdg_error("FaceProlongator error: bad mesh distribution. One dimensional mesh has more than two faces shared between processors.");
            #endif
                const int msg_size = _n_var * n_faces_sendrecv;

                auto& channel = channels.at(k);
                
                channel.partner = partner;
                channel.faces_to_send.reshape(n_faces_sendrecv);
                channel.faces_to_recv.reshape(n_faces_sendrecv);
                channel.send_buf.reshape(msg_size);
                channel.recv_buf.reshape(msg_size);

                for (int i = 0; i < n_faces_sendrecv; ++i)
                {
                    channel.faces_to_send(i) = I.at(i).first;
                    channel.faces_to_recv(i) = I.at(i).second;
                }

                MPI_Send_init(channel.send_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, sreq.get()+k);
                MPI_Recv_init(channel.recv_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, rreq.get()+k);

                ++k;
            }
        }
    #endif
    }

    FaceVector::FaceVector(int n_var_, const Mesh2D& mesh, FaceType face_type_, const QuadratureRule * basis)
        : dim{2},
          _n_var{n_var_},
          _n_basis{basis->n * basis->n},
          _n_faces{mesh.n_edges(face_type_)},
          _face_type{face_type_},
          x(_n_basis, _n_var, 2, _n_faces)
    {
    #ifdef WDG_USE_MPI
        if (_face_type == FaceType::INTERIOR)
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            std::map<int, std::vector<std::pair<int,int>>> sr; // indices of edges and sides to send and recv

            for (int e = 0; e < _n_faces; ++e)
            {
                const Edge * edge = mesh.edge(e, _face_type);

                const int el0 = edge->elements[0];
                const int owner0 = mesh.find_element(el0);

                const int el1 = edge->elements[1];
                const int owner1 = mesh.find_element(el1);

                if (owner0 != owner1)
                {
                    const int partner = (owner0 == rank) ? owner1 : owner0;
                    const int s = (owner0 == rank) ? 0 : 1;

                    sr[partner].push_back({s + 2*e, 1-s + 2*e}); // {send, recv}
                }
            }

            const int edge_size = _n_var * _n_basis;
            
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
                channel.faces_to_send.reshape(n_edges_sendrecv);
                channel.faces_to_recv.reshape(n_edges_sendrecv);
                channel.send_buf.reshape(msg_size);
                channel.recv_buf.reshape(msg_size);

                for (int i=0; i < n_edges_sendrecv; ++i)
                {
                    channel.faces_to_send(i) = I[i].first;
                    channel.faces_to_recv(i) = I[i].second;
                }

                MPI_Recv_init(channel.recv_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, rreq.get()+k);
                MPI_Send_init(channel.send_buf.data(), msg_size, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, sreq.get()+k);

                ++k;
            }
        }
    #endif
    }

#ifdef WDG_USE_MPI
    void FaceVector::send_recv() const
    {
        if (_face_type == FaceType::BOUNDARY)
            return;

        // initialize recieves
        int status = MPI_Startall(rreq.size(), rreq.get());
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Startall failed.", status);

        // copy to buffer and send
        const int m = (dim == 1) ? (_n_var) : (_n_var * _n_basis); // number of values on face to copy
        auto uf = reshape(x, m, 2, _n_faces);
        for (auto& channel : channels)
        {
            const int n_faces_to_send = channel.faces_to_send.size();
            auto sbuf = reshape(channel.send_buf, m, n_faces_to_send);

            for (int f = 0; f < n_faces_to_send; ++f)
            {
                const int s = channel.faces_to_send(f) % 2;
                const int e = channel.faces_to_send(f) / 2;

                for (int i = 0; i < m; ++i)
                {
                    sbuf(i, f) = uf(i, s, e);
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
            const int n_faces_to_recv = channel.faces_to_recv.size();
            auto rbuf = reshape(channel.recv_buf, m, n_faces_to_recv);

            for (int f = 0; f < n_faces_to_recv; ++f)
            {
                const int s = channel.faces_to_recv(f) % 2;
                const int e = channel.faces_to_recv(f) / 2;

                for (int i = 0; i < m; ++i)
                {
                    uf(i, s, e) = rbuf(i, f);
                }
            }
        }

        // wait for send (just in case)
        status = MPI_Waitall(sreq.size(), sreq.get(), MPI_STATUSES_IGNORE);
        if (status != MPI_SUCCESS)
            wdg_error("FaceProlongator::action error: MPI_Waitall failed.", status);
    }
#endif

    FEMVector::FEMVector(int n_var_, const Mesh1D& mesh, const QuadratureRule * basis)
        : dim{1},
          _n_var{n_var_},
          _n_basis{basis->n},
          _n_elem{mesh.n_elem()},
          x(_n_var, _n_basis, _n_elem) {}
        
    FEMVector::FEMVector(int n_var_, const Mesh2D& mesh, const QuadratureRule * basis)
        : dim{2},
          _n_var{n_var_},
          _n_basis{basis->n * basis->n},
          _n_elem{mesh.n_elem()},
          x(_n_var, _n_basis, _n_elem) {}
} // namespace dg
