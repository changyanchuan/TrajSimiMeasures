#ifndef __STRTREE_H__
#define __STRTREE_H__

#include <vector>
#include <algorithm>

#include "point.h"
#include "edge.h"
#include "mbr.h"


bool edge_midpoint_x_cmp(Edge* e1, Edge* e2);

bool edge_midpoint_y_cmp(Edge* e1, Edge* e2);

bool pair_point_x_cmp(std::pair<Point*, unsigned int> p1, std::pair<Point*, unsigned int> p2);

bool pair_point_y_cmp(std::pair<Point*, unsigned int> p1, std::pair<Point*, unsigned int> p2);



class STRTree {
public:

    // partition the space by the centriods of the given edges
    static int Partition(std::vector<Edge*>* edges, const int num_slice,
                            std::vector<float>& vec_x_bounding,
                            std::vector<std::vector<float> >& vec_y_bounding);

    // partition the space by the centriods of the given edges
    static int Partition(std::vector<std::pair<Point*, unsigned int> >* points, const int num_slice,
                            std::vector<MBR*>& vec_mbr, 
                            std::vector<std::vector< unsigned int> >& vec_subsets);

};


#endif