#ifndef _DFT_H_
#define _DFT_H_

#include <string>

#include "edge.h"
#include "mbr.h"
#include "traj.h"
#include "rtree.h"
#include "rnode.h"

class DFT {

public:
    DFT(unsigned short rtree_fanout);
    ~DFT();

    double Build(const std::string& filename, const int& n_trajs, int min_trajlen, int max_trajlen);
    
    int Knn(Traj* q, const std::string& measure, int is_gpu, 
            unsigned int k, unsigned int c, 
            std::vector<std::pair<float, unsigned int> >& results);
    
    double Knn(const std::string& filename, const int& n_trajs, 
            const std::string& measure, int is_gpu, 
            int min_trajlen, int max_trajlen, unsigned int k, unsigned int c,
            std::vector< std::vector<std::pair<float, unsigned int> > >& results);

    int Knn_parallel(const std::string& filename, const int& n_trajs, 
                    const std::string& measure, unsigned int k, unsigned int c);
    
    int Knn_bruteforce(Traj* q, unsigned int k);
    
    int Knn_bruteforce(const std::string& filename, const int& n_trajs, unsigned int k);

    int ReadDataset(const std::string& filename, const int& n, 
                    std::vector<Traj*>& t_set);

    bool Verify();

private:

    int Partitioning(std::vector<Edge*>* edges, const int num_partitions, 
                    std::vector<std::vector<MBR*> >& vec_mbrs,
                    std::vector<std::vector<Edges> >& vec_edges_in_par);
    
    int CreateGlobalLocalIndex(std::vector<std::vector<MBR*> >& vec_mbrs,
                    std::vector<std::vector<Edges > >& vec_edges_in_par);

    int CreateSingleIndex(Edges& vec_edges);
    
    unsigned short rtree_fanout;

    unsigned int num_trajs;
    unsigned int num_edges;

    std::vector<Traj*> trajs;
    std::vector<Edge*> edges;

    RTree* global_index;
    std::vector<RTree*> vec_local_index;
    MBR* space_mbr;

};


#endif