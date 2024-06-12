#ifndef _RNODE_H_
#define _RNODE_H_

#include <vector>
#include <roaring/roaring.hh>
#include "mbr.h"
#include "edge.h"

typedef unsigned short RNodeType;
const RNodeType RNodeTypeUndef  = (unsigned short) 0;
const RNodeType RNodeTypeRoot   = (unsigned short) 1;
const RNodeType RNodeTypeInner  = (unsigned short) 2;
const RNodeType RNodeTypeLeaf   = (unsigned short) 4;

class RTree;

class RNode {

public:
    RNode();
    ~RNode();

    bool AddChild(Edge* e, MBR* mbr);
    bool AddChild(RNode* node, MBR* mbr = nullptr);
    size_t Size() const;
    inline bool IsLeaf() const;
    int RangeQuery_pset(MBR* qmbr, roaring::Roaring& cand_trajids);
    int RangeQuery_pset(std::vector<MBR*> qmbrs, roaring::Roaring& prune_trajids);

    bool Verify();

    RNodeType type;
    MBR* self_mbr;
    RNode* parent;

    std::vector<RNode*> childnodes; // RNodeTypeInner
    std::vector<Edge*> edges; // RNodeTypeLeaf
    std::vector<MBR*> mbrs; // RNodeTypeInner RNodeTypeLeaf
    roaring::Roaring trajids;
};


#endif