#ifndef WDG_MPI_HPP
#define WDG_MPI_HPP

#include "wdg_config.hpp"
#include "wdg_error.hpp"

#include <string>
#include <stdexcept>
#include <memory>
#include <iostream>

#ifdef WDG_USE_MPI
namespace dg
{
    /// @brief manages the MPI environment
    class MPIEnv
    {
    public:
        // calls MPI_Init
        inline MPIEnv(int& argc, char **& argv)
        {
            MPI_Init(&argc, &argv);
        }

        // calls MPI_Finalize
        ~MPIEnv()
        {
            MPI_Finalize();
        }
    };

    // array of MPI_Request that that frees pointers on deletion.
    class RequestVec
    {
    private:
        int n_req;
        std::unique_ptr<MPI_Request[]> req;

    public:
        RequestVec() : n_req{0} {}

        void init(int n)
        {
            n_req = n;
            req.reset(new MPI_Request[n_req]);
        }
        
        ~RequestVec()
        {
            for (int i = 0; i < n_req; ++i)
            {
                MPI_Request_free(req.get() + i);
            }
        }

        int size() const
        {
            return n_req;
        }

        MPI_Request * get()
        {
            return req.get();
        }
    };
} // namespace dg
#endif

#endif