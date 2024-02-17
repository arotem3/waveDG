#include "wdg_error.hpp"

namespace dg
{
    void wdg_error(const std::string& msg, int status)
    {
    #ifdef WDG_USE_MPI
        std::cerr << "\n-------------------\n"
                  << msg << "\n"
                  << "MPI status: " << status << "\n"
                  << "-------------------\n";
        #ifdef WDG_DEBUG
        throw std::runtime_error(msg);
        #else
        MPI_Abort(MPI_COMM_WORLD, 0);
        #endif
    #else
        throw std::runtime_error(msg);
    #endif
    }
} // namespace dg
