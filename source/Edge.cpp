#include "Edge.hpp"

namespace dg
{
#ifdef WDG_USE_MPI
    void StraightEdge::serialize(util::Serializer& serializer) const
    {
        serializer.types.push_back(0);

        int start = serializer.offsets_int.back();
        serializer.offsets_int.push_back(start+7);

        serializer.data_int.push_back(type);
        serializer.data_int.push_back(id);
        serializer.data_int.push_back(elements[0]);
        serializer.data_int.push_back(elements[1]);
        serializer.data_int.push_back(sides[0]);
        serializer.data_int.push_back(sides[1]);
        serializer.data_int.push_back(delta);

        start = serializer.offsets_double.back();
        serializer.offsets_double.push_back(start+7);

        serializer.data_double.push_back(n[0]);
        serializer.data_double.push_back(n[1]);
        serializer.data_double.push_back(meas);
        serializer.data_double.push_back(x[0]);
        serializer.data_double.push_back(x[1]);
        serializer.data_double.push_back(dx[0]);
        serializer.data_double.push_back(dx[1]);
    }

    StraightEdge::StraightEdge(const int *data_int, const double *data_double)
    {
        type = EdgeType(data_int[0]);
        id = data_int[1];
        elements[0] = data_int[2];
        elements[1] = data_int[3];
        sides[0] = data_int[4];
        sides[1] = data_int[5];
        delta = data_int[6];

        n[0] = data_double[0];
        n[1] = data_double[1];
        meas = data_double[2];
        x[0] = data_double[3];
        x[1] = data_double[4];
        dx[0] = data_double[5];
        dx[1] = data_double[6];
    }
#endif
} // namespace dg
