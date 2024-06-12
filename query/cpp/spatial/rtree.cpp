#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <roaring/roaring.hh>

#include "rnode.h"
#include "rtree.h"
#include "mbr.h"
#include "sfc.h"


RTree::RTree(const unsigned short& fanout) : fanout(fanout) {
    this->root = nullptr;
    this->root_mbr = nullptr;
    this->height = 0;
    this->size = 0;
    this->number_of_nodes = 0;
}


RTree::~RTree() {
    delete this->root;
    this->root = nullptr;
    this->root_mbr = nullptr;
}


int RTree::Build(std::vector<Edge*>& input_edges) {
    if (input_edges.size() == 0)
        {return -1;}

    if (this->root)
        {return -2;}

    { // init root and root_mbr
        this->root = new RNode();
        this->root_mbr = this->root->self_mbr;
        this->root->type = RNodeTypeRoot + RNodeTypeLeaf;
        this->number_of_nodes = 1;
        this->height = 1;
    }

    for (auto e: input_edges) {
        Insert(e);
    }

    return 0;
}


int RTree::BulkLoad(std::vector<Edge*>& input_data) {
    return -1;
}


int RTree::Insert(Edge* e) {
    if (!e) 
        {return -1;}
    
    MBR *mbr = new MBR(*e);
    RNode* node2 = nullptr;

    RNode* leaf = ChooseLeaf(mbr);
    
    leaf->AddChild(e, mbr);
    this->size += 1;
    if (leaf->Size() >= this->fanout) { // first add, then check whether full. If yes, split.
        node2 = SplitNode(leaf);
    }
    AdjustTree(leaf, node2, mbr);
    return 0;
}


RNode* RTree::ChooseLeaf(MBR* e_mbr) {
    RNode* node = this->root;

    while(true) {
        if ((node->type & RNodeTypeLeaf) == RNodeTypeLeaf) {
            return node;
        }
        else { // inner node or real root
            float extended_area;
            float extended_area_min = std::numeric_limits<float>::max();
            RNode* node_min = nullptr;

            size_t _i = 0;
            for (auto& mbr: node->mbrs) {
                extended_area = mbr->UnionArea(e_mbr) - mbr->Area();
                if (extended_area < extended_area_min) {
                    extended_area_min = extended_area;
                    node_min = node->childnodes[_i];
                }
                _i += 1;
            }
            node = node_min;
        }
    }
}


RNode* RTree::SplitNode(RNode* node) {
    if (!node)
        {return nullptr;}

    RNode* node2 = new RNode();
    node2->type = node->type;
    node2->parent = node->parent;

    SplitNodeQuadratic(node, node2);

    this->number_of_nodes += 1;

    return node2;
}



int RTree::SplitNodeQuadratic(RNode* node, RNode* node2) {
    // see the origiinal r-tree paper for reference.
    // 1. pick first item for each group
    // 2. add the selected item to each group
    // 3. fill the node with remaining items; size(node) >= fanout / 2

    // 1
    size_t wasted_i, wasted_j;
    {   
        double wasted_area_max = std::numeric_limits<float>::lowest();
        MBR* mbr_i, *mbr_j;
        for (size_t i = 0; i < node->Size() - 1; ++i) {
            mbr_i = node->mbrs[i];
            for (size_t j = i + 1; j < node->Size(); ++j) {
                mbr_j = node->mbrs[j];
                float wasted_area = mbr_i->UnionArea(mbr_j) - mbr_i->Area() - mbr_j->Area();
                if (wasted_area > wasted_area_max) {
                    wasted_area_max = wasted_area;
                    wasted_i = i;
                    wasted_j = j;
                }
            }
        }
    }

    // 2
    std::vector<MBR*> temp_mbrs;
    temp_mbrs.insert(temp_mbrs.end(), node->mbrs.begin(), node->mbrs.end());
    node->mbrs.clear();
    std::vector<Edge*> temp_edges;
    temp_edges.insert(temp_edges.end(), node->edges.begin(), node->edges.end());
    node->edges.clear();
    std::vector<RNode*> temp_childnodes;
    temp_childnodes.insert(temp_childnodes.end(), node->childnodes.begin(), node->childnodes.end());
    node->childnodes.clear();
    
    node->self_mbr->Reset();

    bool is_leaf = (node->type & RNodeTypeLeaf) == RNodeTypeLeaf;

    if (is_leaf) {
        node->AddChild(temp_edges[wasted_i], temp_mbrs[wasted_i]);
        node2->AddChild(temp_edges[wasted_j], temp_mbrs[wasted_j]);
    }
    else {
        node->AddChild(temp_childnodes[wasted_i], temp_mbrs[wasted_i]);
        node2->AddChild(temp_childnodes[wasted_j], temp_mbrs[wasted_j]);
    }
    node->self_mbr->Extend(*temp_mbrs[wasted_i]);
    node2->self_mbr->Extend(*temp_mbrs[wasted_j]);

    if (wasted_i > wasted_j) {
        std::swap(wasted_i, wasted_j);
    }
    temp_mbrs.erase(temp_mbrs.begin() + wasted_j);
    temp_mbrs.erase(temp_mbrs.begin() + wasted_i);

    if (is_leaf) {
        temp_edges.erase(temp_edges.begin() + wasted_j);
        temp_edges.erase(temp_edges.begin() + wasted_i);
    }
    else {
        temp_childnodes.erase(temp_childnodes.begin() + wasted_j);
        temp_childnodes.erase(temp_childnodes.begin() + wasted_i);
    }

    // 3
    {
        while(node->Size() < this->fanout / 2 && node2->Size() < this->fanout / 2) {
            float enlarged_area_max = std::numeric_limits<float>::lowest();
            std::vector<MBR*>::iterator enlarged_it;
            for (auto it = temp_mbrs.begin(); it != temp_mbrs.end(); ++it) {
                float enlarged_area = abs(abs(node->self_mbr->UnionArea(*it) - node->self_mbr->Area()) - 
                        abs(node2->self_mbr->UnionArea(*it) - node2->self_mbr->Area()));
                if (enlarged_area > enlarged_area_max) {
                    enlarged_area_max = enlarged_area;
                    enlarged_it = it;
                }
            }

            if (node->self_mbr->UnionArea(*enlarged_it) - node->self_mbr->Area() 
                    < node2->self_mbr->UnionArea(*enlarged_it) - node2->self_mbr->Area()) {
                if (is_leaf) 
                    node->AddChild( temp_edges[enlarged_it - temp_mbrs.begin()], *enlarged_it  );
                else
                    node->AddChild( temp_childnodes[enlarged_it - temp_mbrs.begin()], *enlarged_it  );
                node->self_mbr->Extend(*(*enlarged_it));
            }
            else {
                if (is_leaf) 
                    node2->AddChild( temp_edges[enlarged_it - temp_mbrs.begin()], *enlarged_it  );
                else
                    node2->AddChild( temp_childnodes[enlarged_it - temp_mbrs.begin()], *enlarged_it  );
                node2->self_mbr->Extend(*(*enlarged_it));
            }
            if (is_leaf)
                temp_edges.erase( temp_edges.begin() + (enlarged_it - temp_mbrs.begin()) );
            else
                temp_childnodes.erase( temp_childnodes.begin() + (enlarged_it - temp_mbrs.begin()) );
            temp_mbrs.erase(enlarged_it);
        }


        auto node_for_remain = node;
        if (node->Size() == this->fanout / 2) { // add remaining to node2
            node_for_remain = node2;
        }
        for (auto it = temp_mbrs.begin(); it < temp_mbrs.end(); ++it) {
            if (is_leaf) 
                node_for_remain->AddChild( temp_edges[it - temp_mbrs.begin()], *it );
            else
                node_for_remain->AddChild( temp_childnodes[it - temp_mbrs.begin()], *it );
            node_for_remain->self_mbr->Extend(*(*it));
        }
        temp_mbrs.clear();
        temp_edges.clear();
        temp_childnodes.clear();
    }

    return 0;
}


int RTree::AdjustTree(RNode* node1, RNode* node2, MBR* mbr) {
    if (node2) {
        if (node1 != this->root) {
            RNode* parent = node1->parent;
            parent->AddChild(node2, node2->self_mbr);
            if (parent->Size() >= this->fanout) { // split
                RNode* parent2 = SplitNode(parent);
                AdjustTree(parent, parent2, mbr);
            }
            else {
                AdjustTree(parent, nullptr, mbr);
            }
        }
        else {
            RNode* newroot = new RNode();
            newroot->type = RNodeTypeRoot;
            newroot->AddChild(node1, node1->self_mbr);
            newroot->self_mbr->Extend(*node1->self_mbr);
            newroot->AddChild(node2, node2->self_mbr);
            newroot->self_mbr->Extend(*node2->self_mbr);
            node1->type = (node1->type & RNodeTypeLeaf) == RNodeTypeLeaf ? RNodeTypeLeaf : RNodeTypeInner;
            node2->type = node1->type;
            node1->parent = newroot;
            node2->parent = newroot;
            this->root = newroot;
            this->root_mbr = newroot->self_mbr;
            
            this->height += 1;
            this->number_of_nodes += 1;
        }
    }
    else {
        node1->self_mbr->Extend(*mbr);
        if (node1 != this->root) {
            AdjustTree(node1->parent, nullptr, mbr);
        }
    }
    return 0;
}


roaring::Roaring RTree::UpdateTrajids(RNode* node) {
    // update trajid list via DFS
    roaring::Roaring r;
    bool is_leaf = (node->type & RNodeTypeLeaf) == RNodeTypeLeaf;
    
    if (is_leaf) {
        for (auto e: node->edges) {
            r.add(e->trajid);
        }
    }
    else { // non-leaf node 
        for (auto n: node->childnodes) {
            r |= this->UpdateTrajids(n);
        }
    }

    node->trajids = r; // copy
    return r;
}


int RTree::RangeQuery_pset(std::vector<MBR*> qtraj, roaring::Roaring& cand_trajids) {
    
    this->root->RangeQuery_pset(qtraj, cand_trajids);
    return 0;
}

bool RTree::Verify() {
    return this->root->Verify();
}


int RTree::SortDataByZRank(std::vector<Edge*>& input_data) {
    
    // sort x
    // scan and map x to a new struct and fill in x ranking
    // sort y (new struct)
    // scan and fill in y rank, computer SFC_Z value
    // sort sfc_z
    // shrink new struct into original edge
    // bulkload bottom up

    class ExtendedEdge {
    public:
        ExtendedEdge(Edge* e) : edge(e), x_rank(0), y_rank(0), sfc_value(0) {}
        ~ExtendedEdge() {this->edge = nullptr;}
        Edge* edge;
        unsigned long x_rank;
        unsigned long y_rank;
        __uint128_t sfc_value;
    };

    auto op_x_sort = [] (Edge* a, Edge*b) {
        return (a->MidPoint().x < b->MidPoint().x
                || ( a->MidPoint().x == b->MidPoint().x 
                    && a->MidPoint().y < b->MidPoint().y ));
    };

    auto op_y_sort = [] (const ExtendedEdge& a, const ExtendedEdge& b) {
        return (a.edge->MidPoint().y < b.edge->MidPoint().y
                || ( a.edge->MidPoint().y == b.edge->MidPoint().y
                    && a.edge->MidPoint().x < b.edge->MidPoint().x  ));
    };

    auto op_sfc_sort = [] (const ExtendedEdge& a, const ExtendedEdge& b) {
        return a.sfc_value < b.sfc_value;
    };

    // sort x
    std::sort(input_data.begin(), input_data.end(), op_x_sort);
    std::vector<ExtendedEdge>* extended_data = new std::vector<ExtendedEdge>();
    {
        int _i = 1;
        for (Edge* _e: input_data) {
            ExtendedEdge _ee(_e);
            _ee.x_rank = _i++;
            extended_data->push_back(_ee);
        }
    }

    // sort y & compute sfc_z
    std::sort(extended_data->begin(), extended_data->end(), op_y_sort);
    {
        long _bits = (long)(log2(extended_data->size()) + 1);

        int _i = 1;
        for (ExtendedEdge& _ee: *extended_data) {
            _ee.y_rank = _i++;
            _ee.sfc_value = SFC_Z(_ee.edge->MidPoint().x, 
                    _ee.edge->MidPoint().y, _bits);
        }
    }

    // sort sfc_z & assign back to the input vector
    std::sort(extended_data->begin(), extended_data->end(), op_sfc_sort);

    {
        std::vector<Edge*>::iterator _e_iter = input_data.begin();
        for(const ExtendedEdge& _ee: *extended_data) {
            *_e_iter = _ee.edge;
            _e_iter++;
        }
    }
    extended_data->clear();

    return 0;
}


std::ostream& operator<<(std::ostream& out, RTree& rtree) {
    out << "RTree.fanout=" << rtree.fanout << "\n"
            << "RTree.height=" << rtree.height << "\n"
            << "RTree.#size=" << rtree.size << "\n"
            << "RTree.#nodes=" << rtree.number_of_nodes << "\n";
    return out;
}


