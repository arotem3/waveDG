#ifndef WDG_SERIALIZER_HPP
#define WDG_SERIALIZER_HPP

#include <vector>

#include "wdg_config.hpp"
#include "MPI_Base.hpp"

#ifdef WDG_USE_MPI
namespace dg
{
    namespace util
    {
        /// @brief used to represent a flattened collection of abstract types that are
        /// composed of numeric data (int or double).
        class Serializer
        {
        private:
            mutable int n[5];

        public:
            std::vector<int> types;

            std::vector<int> offsets_int;
            std::vector<int> data_int;
            
            std::vector<int> offsets_double;
            std::vector<double> data_double;

            Serializer() : offsets_int(1, 0), offsets_double(1, 0.0) {}
            ~Serializer() = default;

            /// sends serialized data over comm to dest. Nonblocking. Sets the range [req, req+6) and returns req+6.
            MPI_Request* send(MPI_Request* req, MPI_Comm comm, int dest, int tag_start=0) const;

            /// recieves serialized data over comm from source. MAY BE BLOCKING:
            /// first a blocking recv call is made to get the size of the data.
            /// Once this message is recieved the data is recieved via irecv i.e.
            /// non-blocking. Sets the range [req, req+5) and returns req+5.
            MPI_Request* recv(MPI_Request* req, MPI_Comm comm, int source, int tag_start=0);
        };
    } // namespace util
} // namespace dg
#endif

#endif