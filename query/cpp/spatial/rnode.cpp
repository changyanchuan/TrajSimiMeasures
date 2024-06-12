#include <cmath>
#include <limits>
#include <iostream>
#include "rnode.h"

RNode::RNode() : type(RNodeTypeUndef) {
    this->self_mbr = new MBR();
    this->parent = nullptr;
}


RNode::~RNode() {
    this->self_mbr = nullptr;
    this->parent = nullptr;

    for (auto& mbr: this->mbrs) 
        delete mbr;

    if ( (this->type & RNodeTypeLeaf) != RNodeTypeLeaf ) {
        for (auto& node: this->childnodes) 
            delete node;
    }
}


bool RNode::AddChild(Edge* e, MBR* mbr) {
    // check whether it is a full node by caller
    this->edges.push_back(e);
    if (!mbr) {
        mbr = new MBR(*e);
    }
    this->mbrs.push_back(mbr);
    return true;
}


bool RNode::AddChild(RNode* node, MBR* mbr) {
    // check whether it is a full node by caller
    this->childnodes.push_back(node);
    node->parent = this;

    if (mbr)
        this->mbrs.push_back(mbr);
    else
        this->mbrs.push_back(node->self_mbr);

    return true;
}


size_t RNode::Size() const {
    return this->mbrs.size();
}


bool RNode::IsLeaf() const {
    return (this->type & RNodeTypeLeaf) == RNodeTypeLeaf;
}


int RNode::RangeQuery_pset(MBR* embr, roaring::Roaring& cand_trajids) {
    
    for (size_t i = 0; i < this->mbrs.size(); ++i) {
        auto mbr = this->mbrs[i];
        bool is_leaf = (this->type & RNodeTypeLeaf) == RNodeTypeLeaf;
        if ( mbr->Intersect(*embr) ) {
            if ( !is_leaf ) {
                this->childnodes[i]->RangeQuery_pset(embr, cand_trajids);
            }
            else {
                cand_trajids.add(this->edges[i]->trajid);
            }
        }
    }
    return 0;
}


int RNode::RangeQuery_pset(std::vector<MBR*> qmbrs, roaring::Roaring& prune_trajids) {
    
    for (size_t i = 0; i < this->mbrs.size(); ++i) {
        bool is_intersect = false;
        for (auto& qmbr: qmbrs) {
            if ( this->mbrs[i]->Intersect(*qmbr) ) {
                is_intersect = true;
                // if ( !this->IsLeaf() && !this->childnodes[i]->IsLeaf() ) {
                if ( !this->IsLeaf() ) {
                    this->childnodes[i]->RangeQuery_pset(qmbrs, prune_trajids);
                }
                else {
                }
                break;
            }
        }
        if (!is_intersect) {
            if ( !this->IsLeaf())
                prune_trajids |= this->childnodes[i]->trajids;
            else
                prune_trajids.add(this->edges[i]->trajid);
        }
    }
    return 0;
}


bool RNode::Verify() {
    bool is_leaf = (this->type & RNodeTypeLeaf) == RNodeTypeLeaf;
    
    for (size_t i = 0; i < this->mbrs.size(); ++i) {
        auto mbr = this->mbrs[i];
        MBR* mbr_tmp = nullptr;
        if (is_leaf) {
            mbr_tmp = new MBR(*(this->edges[i]));
        }
        else {
            mbr_tmp = this->childnodes[i]->self_mbr;
        }
        if (mbr_tmp->xlo != mbr->xlo
                || mbr_tmp->xhi != mbr->xhi
                || mbr_tmp->ylo != mbr->ylo
                || mbr_tmp->yhi != mbr->yhi) {
            std::cerr << "err1" << std::endl;
        }
        if ( is_leaf ) {
            delete mbr_tmp;
        }
        
    }


    MBR* mbr_tmp = new MBR();
    for (size_t i = 0; i < this->mbrs.size(); ++i) {
        mbr_tmp->Extend(*(this->mbrs[i]));
    }
    if (mbr_tmp->xlo != this->self_mbr->xlo
                    || mbr_tmp->xhi != this->self_mbr->xhi
                    || mbr_tmp->ylo != this->self_mbr->ylo
                    || mbr_tmp->yhi != this->self_mbr->yhi) {
        std::cerr << "err2" << " " 
                    << *mbr_tmp << " " 
                    << *this->self_mbr << std::endl;

    }

    roaring::Roaring r;
    if (is_leaf) {
        for (auto e: this->edges) {
            r.add(e->trajid);
        }
    }
    else {
        for (auto c: this->childnodes) {
            r |= c->trajids;
        }
    }
    if (  !((r & this->trajids) == (r | this->trajids)) ) {
        std::cerr << "err3" << std::endl;
    }


    if (!is_leaf) {
        for (auto& c: this->childnodes) {
            if (c->parent != this) {
                std::cerr << "err4" << std::endl;
            }
        }
    }

    if (!is_leaf) {
        for (auto& c: this->childnodes) {
            c->Verify();
        }
    }
    return true;
}

