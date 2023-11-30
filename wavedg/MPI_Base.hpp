#ifndef WDG_MPI_HPP
#define WDG_MPI_HPP

#include "wdg_config.hpp"
#include <string>
#include <stdexcept>
#include <memory>

namespace dg
{
    inline void mpi_error_and_abort_on_fail(const std::string& fun, int status, const std::string& msg="")
    {
        if (status != MPI_SUCCESS)
        {
            std::cerr << fun << " failed and returned status: " << status << "\n";

            if (not msg.empty())
            {
                std::cerr << msg << std::endl;
            }

            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // array of MPI_Request that that frees pointers on deletion.
    class RequestVec
    {
    private:
        int n_req;
        std::unique_ptr<MPI_Request[]> req;

    public:
        RequestVec() {}

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