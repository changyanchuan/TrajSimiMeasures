#ifndef _DITA_H_
#define _DITA_H_

#include <string>

#include "edge.h"
#include "mbr.h"
#include "traj.h"
#include "rtree.h"
#include "rnode.h"

class DITA {

public:
    DITA(unsigned short rtree_fanout, unsigned short num_pivots);
    ~DITA();

    double Build(const std::string& filename, const int& n_trajs, int min_trajlen, int max_trajlen);
    
    double Knn(const std::string& filename, const int& n_trajs, const std::string& measure, 
            int is_gpu, int min_trajlen, int max_trajlen, unsigned int k, unsigned int c,
            std::vector< std::vector<std::pair<float, unsigned int> > >& results);

private:
    int ConstructIndexLevel(RNode* treenode, int level, 
                            std::vector<std::pair<Points, Traj*> >& vec_vpoints,
                            std::vector<unsigned int>& selected_indices);
    int CreatePivotTrajs(std::vector<Traj*>& vec_traj, std::vector<std::pair<Points, Traj*> >& vec_vpoints);

    int SelectPivot(Traj* traj, Points& pivots);

    int Partitioning(std::vector<std::pair<Points, Traj*> >& vec_vpoints, 
                        std::vector<unsigned int>& selected_indices,
                        int level,
                        const int num_partitions,
                        // std::vector<float>& vec_x_bounding, 
                        // std::vector<std::vector<float> >& vec_y_bounding,
                        // std::vector<std::vector< std::vector<unsigned int> > >& vec_idx_in_par);
                        std::vector<MBR*>& vec_mbr, 
                        std::vector<std::vector< unsigned int> >& vec_subsets);
    
    int Knn_bruteforce(Traj* q, const std::string& measure, unsigned int k);
    int Knn(Traj* q, const std::string& measure, int is_gpu, unsigned int k, unsigned int c, 
            std::vector<std::pair<float, unsigned int> >& results);
    
    int KnnByLevelFn(Traj* q, int level, RNode* rnode, float tau, roaring::Roaring& trajids_candidates);
   

    unsigned short rtree_fanout;
    unsigned short num_pivots; // > 0

    unsigned int num_trajs;

    std::vector<Traj*> trajs;
    std::vector<std::pair<Points, Traj*> > pivots;

    RTree* index;
    MBR* space_mbr;

};


#endif