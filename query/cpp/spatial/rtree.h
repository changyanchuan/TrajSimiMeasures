#ifndef __RTREE_H__
#define __RTREE_H__

#include <vector>
#include <queue>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <string>
#include <roaring/roaring.hh>

#include "traj.h"
#include "edge.h"
#include "rnode.h"


// rtree with bulkloaded construction
class RTree {

public:
    RTree(const unsigned short& fanout);
    ~RTree();

    int Build(std::vector<Edge*>& input_data);
    int BulkLoad(std::vector<Edge*>& input_data);
    int Insert(Edge* e);
    int RangeQuery_pset(std::vector<MBR*> qtraj, roaring::Roaring& cand_trajids);
    roaring::Roaring UpdateTrajids(RNode* node);
    bool Verify();

    RNode* root;
    MBR* root_mbr;
    unsigned short fanout;
    unsigned short height;
    unsigned long size;
    unsigned long number_of_nodes;
    
    friend std::ostream& operator<<(std::ostream& out, RTree& rtree);

private:
    RNode* ChooseLeaf(MBR* e_mbr);
    RNode* SplitNode(RNode* node);
    int SplitNodeQuadratic(RNode* node, RNode* node2);
    int AdjustTree(RNode* node1, RNode* node2, MBR* mbr);

    int SortDataByZRank(std::vector<Edge*>& input_data);
};


#endif