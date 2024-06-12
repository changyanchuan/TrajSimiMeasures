
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

#include "strtree.h"
#include "point.h"
#include "edge.h"
#include "mbr.h"


bool edge_midpoint_x_cmp(Edge* e1, Edge* e2) {
    return e1->MidPoint().x < e2->MidPoint().x;
}

bool edge_midpoint_y_cmp(Edge* e1, Edge* e2) {
    return e1->MidPoint().y < e2->MidPoint().y;
}

bool pair_point_x_cmp(std::pair<Point*, unsigned int> p1, std::pair<Point*, unsigned int> p2) {
    return (p1.first)->x < (p2.first)->x;
}

bool pair_point_y_cmp(std::pair<Point*, unsigned int> p1, std::pair<Point*, unsigned int> p2) {
    return (p1.first)->y < (p2.first)->y;
}

int STRTree::Partition(std::vector<Edge*>* edges, const int num_slice,
                            std::vector<float>& vec_x_bounding,
                            std::vector<std::vector<float> >& vec_y_bounding) {
    // dont return list of MBR
    // because edges are a subset, the caller needs to refine the mbr ( to extend the boundary )
    // return slice location is better. 

    int n = edges->size();
    int slice_capacity = (int)(n / num_slice);
    int slice_capacity_reminder = n % num_slice;

    std::sort(edges->begin(), edges->end(), edge_midpoint_x_cmp);
    
    // sort along x then along y,
    // then partition
    for (int i = 0; i < num_slice; ++i) {
        size_t i_start, i_end;
        if (i < slice_capacity_reminder) {
            i_start = i * (slice_capacity + 1);
            i_end = i_start + (slice_capacity + 1) - 1; // included, dont minus 1
        }
        else {
            i_start = slice_capacity_reminder * (slice_capacity + 1) + (i - slice_capacity_reminder) * slice_capacity;
            i_end = i_start + slice_capacity - 1;
        }

        if (i == 0) {
            vec_x_bounding.push_back( (*edges)[i_start]->MidPoint().x );
        }
        vec_x_bounding.push_back( (*edges)[i_end]->MidPoint().x );

        std::sort(edges->begin() + i_start, edges->begin() + i_end + 1, edge_midpoint_y_cmp);

        int n_in_slice = i_end - i_start + 1; // num of elements in a slice
        // int slice_sub_capacity = (int) ceil((double)n_in_slice / num_slice); // num of sub-slice in a slice
        int slice_sub_capacity = (int)(n_in_slice / num_slice);
        int slice_sub_capacity_reminder = n_in_slice % num_slice;

        int j = 0; // overall index in a slice
        std::vector<float> _vec_y;
        for (int j = 0; j < num_slice; ++j) {
            size_t j_start, j_end;
            if (j < slice_sub_capacity_reminder) {
                j_start = j * (slice_sub_capacity + 1) + i_start;
                j_end = j_start + (slice_sub_capacity + 1) - 1;
            }
            else {
                j_start = slice_sub_capacity_reminder * (slice_sub_capacity + 1) + (j - slice_sub_capacity_reminder) * slice_sub_capacity + i_start;
                j_end = j_start + slice_sub_capacity - 1;
            }
            if (j == 0) {
                _vec_y.push_back((*edges)[j_start]->MidPoint().y);
            }
            _vec_y.push_back((*edges)[j_end]->MidPoint().y);
        }
        vec_y_bounding.push_back(_vec_y);
    }
    return 0;
}

// partition the space by the centriods of the given edges
int STRTree::Partition(std::vector<std::pair<Point*, unsigned int> >* points, const int num_slice,
                        std::vector<MBR*>& vec_mbr, 
                        std::vector<std::vector< unsigned int> >& vec_subsets) {
                        // std::vector<float>& vec_x_bounding,
                        // std::vector<std::vector<float> >& vec_y_bounding) {
    // dont return list of MBR
    // because edges are a subset, the caller needs to refine the mbr ( to extend the boundary )
    // return slice location is better. 

    int n = points->size();
    int slice_capacity = (int)(n / num_slice);
    int slice_capacity_reminder = n % num_slice;

    std::sort(points->begin(), points->end(), pair_point_x_cmp);
    
    // sort along x then along y,
    // then partition
    for (int i = 0; i < num_slice; ++i) {
        size_t i_start, i_end;
        if (i < slice_capacity_reminder) {
            i_start = i * (slice_capacity + 1);
            i_end = i_start + (slice_capacity + 1) - 1; // included, dont minus 1
        }
        else {
            i_start = slice_capacity_reminder * (slice_capacity + 1) + (i - slice_capacity_reminder) * slice_capacity;
            i_end = i_start + slice_capacity - 1;
        }


        float mbr_xlo = (*points)[i_start].first->x;
        float mbr_xhi = (*points)[i_end].first->x;

        std::sort(points->begin() + i_start, points->begin() + i_end + 1, pair_point_y_cmp);

        int n_in_slice = i_end - i_start + 1; // num of elements in a slice
        // int slice_sub_capacity = (int) ceil((double)n_in_slice / num_slice); // num of sub-slice in a slice
        int slice_sub_capacity = (int)(n_in_slice / num_slice);
        int slice_sub_capacity_reminder = n_in_slice % num_slice;

        int j = 0; // overall index in a slice
        std::vector<float> _vec_y;
        for (int j = 0; j < num_slice; ++j) {
            size_t j_start, j_end;
            if (j < slice_sub_capacity_reminder) {
                j_start = j * (slice_sub_capacity + 1) + i_start;
                j_end = j_start + (slice_sub_capacity + 1) - 1;
            }
            else {
                j_start = slice_sub_capacity_reminder * (slice_sub_capacity + 1) + (j - slice_sub_capacity_reminder) * slice_sub_capacity + i_start;
                j_end = j_start + slice_sub_capacity - 1;
            }

            MBR* mbr = new MBR(mbr_xlo, mbr_xhi,
                                (*points)[j_start].first->y, (*points)[j_end].first->y);
            vec_mbr.push_back(mbr);
            std::vector<unsigned int> subset_indices;
            for (int jj = j_start; jj <= j_end; ++jj) {
                subset_indices.push_back((*points)[jj].second);
            }
            vec_subsets.push_back(subset_indices);
        }
    }
    return 0;
}

