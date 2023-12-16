#ifndef WDG_ERROR_HPP
#define WDG_ERROR_HPP

#include <string>
#include <exception>
#include <iostream>

#include "wdg_config.hpp"

namespace dg
{
    void wdg_error(const std::string& msg, int status=0);
} // namespace dg

#endif