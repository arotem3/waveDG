#include "Serializer.hpp"

#ifdef WDG_USE_MPI
namespace dg
{
    namespace util
    {
        MPI_Request* Serializer::send(MPI_Request* req, MPI_Comm comm, int dest, int tag) const
        {
            n[0] = types.size();
            n[1] = offsets_int.size();
            n[2] = data_int.size();
            n[3] = offsets_double.size();
            n[4] = data_double.size();

            MPI_Isend(n, 5, MPI_INT, dest, tag, comm, req);
            ++req;

            MPI_Isend(types.data(), n[0], MPI_INT, dest, tag+1, comm, req);
            ++req;

            MPI_Isend(offsets_int.data(), n[1], MPI_INT, dest, tag+2, comm, req);
            ++req;

            MPI_Isend(data_int.data(), n[2], MPI_INT, dest, tag+3, comm, req);
            ++req;

            MPI_Isend(offsets_double.data(), n[3], MPI_INT, dest, tag+4, comm, req);
            ++req;

            MPI_Isend(data_double.data(), n[4], MPI_DOUBLE, dest, tag+5, comm, req);
            ++req;

            return req;
        }

        MPI_Request* Serializer::recv(MPI_Request* req, MPI_Comm comm, int source, int tag)
        {
            int status = MPI_Recv(n, 5, MPI_INT, source, tag, comm, MPI_STATUS_IGNORE);
            WDG_MPI_CHECK_SUCCESS("MPI_Recv", status);
            
            types.resize(n[0]);
            offsets_int.resize(n[1]);
            data_int.resize(n[2]);
            offsets_double.resize(n[3]);
            data_double.resize(n[4]);

            MPI_Irecv(types.data(), n[0], MPI_INT, source, tag+1, comm, req);
            ++req;

            MPI_Irecv(offsets_int.data(), n[1], MPI_INT, source, tag+2, comm, req);
            ++req;

            MPI_Irecv(data_int.data(), n[2], MPI_INT, source, tag+3, comm, req);
            ++req;

            MPI_Irecv(offsets_double.data(), n[3], MPI_INT, source, tag+4, comm, req);
            ++req;

            MPI_Irecv(data_double.data(), n[4], MPI_DOUBLE, source, tag+5, comm, req);
            ++req;

            return req;
        }
    } // namespace util
} // namespace dg
#endif
