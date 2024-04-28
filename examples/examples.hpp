#ifndef WDG_EXAMPLES_HPP
#define WDG_EXAMPLES_HPP

#include <iostream>
#include <fstream>
#include <iomanip>

namespace dg
{
    /// @brief save array as raw binary to file
    /// @param fname file name to save to
    /// @param n length of u
    /// @param u array to save
    inline void to_file(const std::string& fname, int n, const double * u)
    {
        std::ofstream out(fname, std::ios::out | std::ios::binary);
        out.write(reinterpret_cast<const char*>(u), n * sizeof(double));
        out.close();
    }

    /// @brief Used to print progress bar.
    class ProgressBar
    {
    public:
        /// @brief initialize progress bar
        /// @param max_iterations maximum number of iterations such that after
        /// max_iteration increments, the progress bar is full.
        inline ProgressBar(int max_iterations) : it{0}, nt{max_iterations}, progress(30, ' ') {}

        inline ProgressBar& operator++()
        {
            it = std::min(it+1, nt-1);
            for (int i=0; nt*i < 30*(it-1); ++i)
                progress.at(i) = '#';
            return *this;
        }

        inline const std::string& get() const
        {
            return progress;
        }

    private:
        int it;
        const int nt;
        std::string progress;
    };
} // namespace dg

#endif
