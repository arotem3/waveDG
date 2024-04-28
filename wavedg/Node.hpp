#ifndef WDG_NODES_HPP
#define WDG_NODES_HPP

#include <vector>

namespace dg
{
    enum class NodeType
    {
        INTERIOR,
        BOUNDARY
    };

    struct Node
    {
        struct element_info
        {
            int i; // identifies the corner of the element that this node corresponds to, one of [0, 1, 2, 3]
            int id; // global element index
        };

        int id;
        NodeType type;
        double x[2]; // node location
        std::vector<element_info> connected_elements;
    };
    
} // namespace dg


#endif
