#ifndef WDG_MPI_HPP
#define WDG_MPI_HPP

#include "wdg_config.hpp"
#include <string>
#include <stdexcept>

#ifdef WDG_USE_MPI
    #define WDG_MPI_CHECK_SUCCESS(fun, status) \
        if (status != MPI_SUCCESS) \
            throw std::runtime_error(fun " failed and returned status: " + std::to_string(status));
#endif

#endif