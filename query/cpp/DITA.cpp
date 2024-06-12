#include <random>
#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <roaring/roaring.hh> // namespace roaring
#include <iostream>
#include <fstream>
#include <string>
#include <limits>

#include "DITA.h"
#include "strtree.h"
#include "rtree.h"
#include "edge.h"
#include "tool_funcs.h"
#include "BS_thread_pool.hpp"
#include "dataset.hpp"


DITA::DITA(unsigned short rtree_fanout, unsigned short num_pivots) {
    this->num_pivots = num_pivots;
    this->rtree_fanout = rtree_fanout;
    this->space_mbr = new MBR();

    this->index = new RTree(this->rtree_fanout);
    this->index->root = new RNode(); 
    this->index->root->type = RNodeTypeRoot;
    this->index->root->self_mbr = this->space_mbr;
    this->index->number_of_nodes += 1;
}

DITA::~DITA() { 
    delete this->space_mbr;
    this->space_mbr = nullptr;

    delete this->index;

    for (auto& t: this->trajs) {
        delete t;
    }
    this->trajs.clear();
    this->pivots.clear();
}


double DITA::Build(const std::string& filename, const int& n_trajs, 
                int min_trajlen, int max_trajlen) {
    // 1. read all trajs
    // create pivot trajectories
    // generate MBRs for all trajectories
    // do partitioning based on nodes, then create one tree level
    //      recursively create the each level
    // fill the bitmap of trajids bottom-up. may move this to the rtree??

    // 1.
    Dataset::Load(filename, n_trajs, min_trajlen, max_trajlen, 2000, this->trajs); // TODO
    this->num_trajs = this->trajs.size();

    Timer timer_;

    this->CreatePivotTrajs(this->trajs, this->pivots);

    MBR* mbr = new MBR();
    for (auto pp: this->pivots) {
        auto p = pp.first[0];
        if (p->x < mbr->xlo) mbr->xlo = p->x;
        if (p->x > mbr->xhi) mbr->xhi = p->x;
        if (p->y < mbr->ylo) mbr->ylo = p->y;
        if (p->y > mbr->yhi) mbr->yhi = p->y;
    }
    this->space_mbr->SetMBR(mbr->xlo, mbr->xhi, mbr->ylo, mbr->yhi);
    std::cout << "[Dataset]#trajs" << this->pivots.size() << "/trajs=" << this->trajs.size() << std::endl;
    std::cout << "[Dataset]mbr=" << *(this->space_mbr) << std::endl;

    std::vector<unsigned int> selected_indices(this->num_trajs);
    std::iota(selected_indices.begin(), selected_indices.end(), 0);
    this->ConstructIndexLevel(this->index->root, 0, this->pivots, selected_indices);

    double buildtime = timer_.Elapse();
    std::cout << "[DFT::Build]done. @=" << buildtime << 
                    ", #height=" << this->index->height << 
                    ", #nodes=" << this->index->number_of_nodes << std::endl;

    return buildtime;
}

int DITA::ConstructIndexLevel(RNode* treenode, int level, 
                            std::vector<std::pair<Points, Traj*> >& vec_vpoints,
                            std::vector<unsigned int>& selected_indices) {
    
    if (level + 1 > this->index->height) {
        this->index->height = level + 1;
    }
    
    for (auto idx: selected_indices) {
        treenode->trajids.add( vec_vpoints[idx].second->id );
    }

    if (selected_indices.size() <= this->rtree_fanout || level == this->num_pivots+1) {
        treenode->type = RNodeTypeLeaf;
        return 0;
    }
    else {
        treenode->type = RNodeTypeInner;
    }


    std::vector<MBR*> vec_mbr;
    std::vector<std::vector< unsigned int> > vec_subsets;

    this->Partitioning(vec_vpoints, selected_indices, level, this->rtree_fanout, 
                        vec_mbr, vec_subsets);

    for (size_t i = 0; i < vec_mbr.size(); ++i) {
        RNode* child = new RNode();
        this->index->number_of_nodes += 1;
        child->self_mbr = vec_mbr[i];
        treenode->AddChild(child);
        auto sub_indices = vec_subsets[i];
        ConstructIndexLevel(child, level + 1, vec_vpoints, sub_indices);
    }

    return 0;
}    



int DITA::CreatePivotTrajs(std::vector<Traj*>& vec_traj, std::vector<std::pair<Points, Traj*> >& vec_vpoints) {

    for(auto t: vec_traj) {
        Points pivots;
        this->SelectPivot(t, pivots);
        vec_vpoints.push_back(std::make_pair(pivots, t));
    }
    return 0;
}


int DITA::SelectPivot(Traj* traj, Points& pivots) {
    // Neighbour Distance Strategy
    
    assert(traj->NumPoints() - 2 >= this->num_pivots);
    auto k = this->num_pivots;
    
    std::vector<std::pair<float, unsigned int> > dists;
    for(size_t i = 0; i < traj->NumPoints() - 2; ++i) {
        float dist = L2Squared(traj->points[i]->x, traj->points[i]->y, 
                                traj->points[i+1]->x, traj->points[i+1]->y);
        dists.push_back( std::make_pair(dist, i+1) );
    }
    std::nth_element(dists.begin(), dists.begin() + k-1, dists.end(), 
                    cmp_pair_1st_reverse<float, unsigned int>);

    std::sort(dists.begin(), dists.begin() + k, cmp_pair_2nd<float, unsigned int>);

    pivots.push_back(traj->points[0]);
    pivots.push_back(traj->points[traj->NumPoints()-1]);

    for (size_t i = 0; i < k; ++i) {
        unsigned int idx = dists[i].second;
        pivots.push_back(traj->points[idx]);
    }

    return 0;
}


// return parameters: vec_edges_in_par
int DITA::Partitioning(std::vector<std::pair<Points, Traj*> >& vec_vpoints, 
                        std::vector<unsigned int>& selected_indices,
                        int level,
                        const int num_partitions,
                        // std::vector<float>& vec_x_bounding, 
                        // std::vector<std::vector<float> >& vec_y_bounding,
                        // std::vector<std::vector< std::vector<std::pair<Point*, Traj*> > > >& vec_edges_in_par) {
                        // std::vector<std::vector< std::vector<unsigned int> > >& vec_idx_in_par) {
                        std::vector<MBR*>& vec_mbr, 
                        std::vector<std::vector< unsigned int> >& vec_subsets) {

    // 1. subsampling
    // 2. do partion; 
    //      it is hard to extract it as a str_partion, since here we use sampled edges. 
    //      the obtained mbr of sampled edges cannot cover the whole space. 
    //      hard to define a general return type
    // 3. refine the boundary of all MBRs; Since the x, y splits currently are based
    //      on the sampled points, the boundary of MBRs are not the boundary of all edges.
    // 5. traverse the segments


    int rtn = 0;

    // float sampling_rate = 0.01; // TODO: move to the argument list?
    // int n_samples = (int)(points->size() * sampling_rate);
    // std::vector<std::pair<Point*, Traj*> >* sample_points = new std::vector<std::pair<Point*, Traj*> >();

    // 1.
    // std::shuffle(points->begin(), points->end(), std::default_random_engine(g_seed));
    // sample_points->assign(points->begin(), points->begin() + n_samples); // shallow copy

    // 2.  
    int num_slice = (int) sqrt( (double)num_partitions ); 
    // std::vector<float> vec_x_bounding;
    // std::vector<std::vector<float> > vec_y_bounding;
    std::vector<std::pair<Point*, unsigned int> > points;
    for (auto idx: selected_indices) {
        points.push_back( std::make_pair( vec_vpoints[idx].first[level], idx ) ) ;
    }
    // STRTree::Partition(&points, num_slice, vec_x_bounding, vec_y_bounding);
    STRTree::Partition(&points, num_slice, vec_mbr, vec_subsets);
    // 3.
    // vec_x_bounding.front() = this->space_mbr->xlo < vec_x_bounding.front() ? this->space_mbr->xlo : vec_x_bounding.front();
    // vec_x_bounding.back() = this->space_mbr->xhi > vec_x_bounding.back() ? this->space_mbr->xhi : vec_x_bounding.back();

    // for (auto& vec_y: vec_y_bounding) {
    //     vec_y.front() = this->space_mbr->ylo < vec_y.front() ? this->space_mbr->ylo : vec_y.front();
    //     vec_y.back() = this->space_mbr->yhi > vec_y.back() ? this->space_mbr->yhi : vec_y.back();
    // }
    /*
    // 5. first x then y
    vec_idx_in_par.clear();
    for (int i = 0; i < vec_y_bounding.size(); ++i) {
        // vec_idx_in_par.push_back( std::vector< std::vector<std::pair<Point*, Traj*> > >() );
        vec_idx_in_par.push_back( std::vector< std::vector<unsigned int> >() );
        for (int j = 0; j < vec_y_bounding[i].size() - 1; ++j) {
            // vec_idx_in_par.back().push_back( std::vector<std::pair<Point*, Traj*> > >() );
            vec_idx_in_par.back().push_back( std::vector<unsigned int>() );
        }
    }

    for (auto pp: points) {
        auto p = pp.first;
        float _x = p->x;
        float _y = p->y;

        int _x_idx = std::lower_bound(vec_x_bounding.begin(), vec_x_bounding.end(), _x) - vec_x_bounding.begin();
        // if (_x != vec_x_bounding[_x_idx] || _x_idx == vec_x_bounding.size() - 1) {
        if (_x_idx > 0) {
            _x_idx -= 1;
        }

        int _y_idx = std::lower_bound(vec_y_bounding[_x_idx].begin(), vec_y_bounding[_x_idx].end(), _y) - vec_y_bounding[_x_idx].begin();
        // if (_y != vec_y_bounding[_x_idx][_y_idx] || _y_idx == vec_y_bounding[_x_idx].size() - 1) {
        if (_y_idx > 0) {
            _y_idx -= 1;
        }
        // vec_edges_in_par[_x_idx][_y_idx].push_back( std::make_pair(p, pp.second) );
        vec_idx_in_par[_x_idx][_y_idx].push_back( pp.second );
    }
    */
    // std::cout << "[DITA::Partitioning]done." << std::endl;
    return rtn;
}


int DITA::Knn_bruteforce(Traj* q, const std::string& measure, unsigned int k) {
    Timer _timer;
    auto vec_dist = new std::vector<std::pair<float, unsigned int> >();
    
    for (auto& t: this->trajs) {
        if (!measure.compare("DTW"))
            vec_dist->push_back( std::make_pair(q->DTW(t), t->id) );
        else if (!measure.compare("ERP"))
            vec_dist->push_back( std::make_pair(q->ERP(t, Traj::erp_gpoint), t->id) );
    }
    std::nth_element(vec_dist->begin(), vec_dist->begin() + k-1, vec_dist->end());
    std::sort(vec_dist->begin(), vec_dist->begin() + k);
    std::cout << "[KnnBF]tid=" << q->id << ",@=" << _timer <<
                        ",k-th_dist=" << (*vec_dist)[k-1].first << std::endl;

    delete vec_dist;
    return 0;
}


int DITA::Knn(Traj* q, const std::string& measure,
                int is_gpu, unsigned int k, unsigned int c, 
                std::vector<std::pair<float, unsigned int> >& results) {

    Timer _timer;

    roaring::Roaring trajids_samples;
    Point* p_1 = q->points[0];
    Point* p_n = q->points[q->NumPoints()-1];

    size_t nearest_mbridx = 0;
    float nearest_mbrdist = std::numeric_limits<float>::max();
    for (auto i = 0; i < this->index->root->Size(); ++i) {
        float mbr_p_dist = this->index->root->mbrs[i]->PointToMBRDistance(*p_1);
        if (mbr_p_dist < nearest_mbrdist) {
            nearest_mbrdist = mbr_p_dist;
            nearest_mbridx = i;
        } 

        // if (this->index->root->mbrs[i]->Contain(*p_1)) {
            // trajids_samples |= this->index->root->childnodes[i]->trajids;
            // RNode* node = this->index->root->childnodes[i];
            // for (auto j = 0; j < node->Size(); ++j) { 
            //     if (node->mbrs[j]->Contain(*p_n)) {
            //         trajids_samples |= node->childnodes[j]->trajids;
            //     }
            // }
        // }
    }
    trajids_samples |= this->index->root->childnodes[nearest_mbridx]->trajids;

    // std::cout << *q << std::endl;
    int n_cand = trajids_samples.cardinality(); // TODO: if #trajids_samples < c*k 
    
    std::vector<unsigned int> vec_sampled_trajs;
    if (n_cand > c*k) {
        auto set_sample_nums = PickSet(n_cand, c*k);
        for (auto i: set_sample_nums) {
            uint32_t trajid;
            trajids_samples.select(i, &trajid);
            vec_sampled_trajs.push_back((unsigned int)trajid);
        }
    }
    else if (n_cand >= k){
        for (int i = 0; i < n_cand; ++i) {
            uint32_t trajid;
            trajids_samples.select(i, &trajid);
            vec_sampled_trajs.push_back((unsigned int)trajid);
        }
    }
    assert(n_cand >= k || !(std::cerr << k << "," << n_cand) );

    
    // trajsimi dist between sampled trajs and q
    std::vector<float> vec_sampled_dist; 
    if (!is_gpu) {
        for (auto& trajid: vec_sampled_trajs) {
            Traj* dtraj = trajs[trajid];
            if (!measure.compare("DTW"))
                vec_sampled_dist.push_back(  dtraj->DTW(q)  );
            else if (!measure.compare("ERP"))
                vec_sampled_dist.push_back(  dtraj->ERP(q, Traj::erp_gpoint)  );
        }
    }
    else {
        std::vector<Traj*> trajs1;
        std::vector<Traj*> trajs2;
        for (auto& trajid: vec_sampled_trajs) {
            trajs1.push_back(trajs[trajid]);
            trajs2.push_back(q);
        }
        Traj::TrajSimiGPU(trajs1, trajs2, measure, vec_sampled_dist);
    }


    std::nth_element(vec_sampled_dist.begin(), vec_sampled_dist.begin() + k-1, vec_sampled_dist.end());
    float epsilon = vec_sampled_dist[k - 1];
    // std::cout << "eps=" << epsilon << std::endl;

    // query knn
    roaring::Roaring trajids_candidates;
    this->KnnByLevelFn(q, 0, this->index->root, epsilon, trajids_candidates);
    g_counter.Add(CNT_KNN_TRAJ_RETRIVED, trajids_candidates.cardinality());

    // refine candidates; optimised verification
    std::vector< std::pair<float, unsigned int> > cand_trajsimi;
    MBR* mbr_q = new MBR(*q);
    MBR* embr_q = new MBR(mbr_q->xlo, mbr_q->xhi, mbr_q->ylo, mbr_q->yhi);
    embr_q->Extend(epsilon);

    if (!is_gpu) {
        for (auto _trajid: trajids_candidates) {
            MBR* mbr_t = new MBR( *(this->trajs[_trajid]) );
            MBR* embr_t = new MBR(mbr_t->xlo, mbr_t->xhi, mbr_t->ylo, mbr_t->yhi);
            embr_t->Extend(epsilon);
            if ( embr_t->Contain(*mbr_q) && embr_q->Contain(*mbr_t) ) {
                float _dist;
                if (!measure.compare("DTW"))
                    _dist = this->trajs[_trajid]->DTW(q);
                else if (!measure.compare("ERP"))
                    _dist = this->trajs[_trajid]->ERP(q, Traj::erp_gpoint);
                cand_trajsimi.push_back(  std::make_pair(_dist, _trajid)  );
            }
            delete mbr_t;
            delete embr_t;
            mbr_t = nullptr;
            embr_t = nullptr;
        }
    }
    else {
        std::vector<Traj*> trajs1;
        std::vector<Traj*> trajs2;
        std::vector<float> dists;

        for (auto _trajid: trajids_candidates) {
            MBR* mbr_t = new MBR( *(this->trajs[_trajid]) );
            MBR* embr_t = new MBR(mbr_t->xlo, mbr_t->xhi, mbr_t->ylo, mbr_t->yhi);
            embr_t->Extend(epsilon);
            if ( embr_t->Contain(*mbr_q) && embr_q->Contain(*mbr_t) ) {
                trajs1.push_back(this->trajs[_trajid]);
                trajs2.push_back(q);
            }
            delete mbr_t;
            delete embr_t;
            mbr_t = nullptr;
            embr_t = nullptr;
        }
        Traj::TrajSimiGPU(trajs1, trajs2, measure, dists);
        for (size_t i = 0; i < dists.size(); ++i) {
            auto _dist = dists[i];
            auto _trajid = trajs1[i]->id;
            cand_trajsimi.push_back(  std::make_pair(_dist, _trajid)  );
        }
        dists.clear();
    }

    std::nth_element(cand_trajsimi.begin(), cand_trajsimi.begin() + k-1,
                    cand_trajsimi.end(), cmp_pair_1st<float, unsigned int>);
    std::sort(cand_trajsimi.begin(), cand_trajsimi.begin() + k, cmp_pair_1st<float, unsigned int>);

    std::cout << "[Knn]tid=" << q->id << ",@=" << _timer <<
                        ",n_samples_in_L1=" << n_cand <<
                        ",epsilon=" << epsilon << 
                        ",k-th_dist=" << cand_trajsimi[k-1].first << 
                        ",#traj_retrived=" << trajids_candidates.cardinality() << 
                        ",#cand_traj=" << cand_trajsimi.size() << std::endl;

    vec_sampled_trajs.clear();
    vec_sampled_dist.clear();
    cand_trajsimi.clear();

    delete mbr_q;
    delete embr_q;

    return 0;
}


int DITA::KnnByLevelFn(Traj* q, int level, RNode* rnode, float tau, 
                        roaring::Roaring& trajids_candidates) {

    if (rnode->type == RNodeTypeLeaf) {
        // TODO
        trajids_candidates |= rnode->trajids;
        return 0;
    }

    if (level <= 1) {
        Point* p = nullptr;
        if (level == 0) {
            p = q->points[0];
        }
        else if (level == 1) {
            p = q->points[q->NumPoints()-1];
        }

        for (size_t i = 0; i < rnode->Size(); ++i) {
            float dist = rnode->mbrs[i]->PointToMBRDistance(*p);
            if (dist <= tau) {
                KnnByLevelFn(q, level + 1, rnode->childnodes[i], tau-dist, trajids_candidates);
            }
        }
    }
    else {
        for (size_t i = 0; i < rnode->Size(); ++i) {
            float dist = rnode->mbrs[i]->TrajToMBRDistance(*q);
            if (dist <= tau) {
                KnnByLevelFn(q, level + 1, rnode->childnodes[i], tau-dist, trajids_candidates);
            }
        }
    }
    
    return 0;
}



double DITA::Knn(const std::string& filename, const int& n_trajs, const std::string& measure, 
                int is_gpu, int min_trajlen, int max_trajlen, unsigned int k, unsigned int c,
                std::vector< std::vector<std::pair<float, unsigned int> > >& results) {
    
    if (!measure.compare("ERP")) {
        float x, y;
        if (filename.find("geolife") != std::string::npos) {
            x = 12848366.496750318;
            y = 4785127.712059225;
        }
        else if (filename.find("porto") != std::string::npos) {
            x = -968535.007007895;
            y = 5027117.916994769;
        }
        else if (filename.find("xian") != std::string::npos) {
            x = 12848366.496750318;
            y = 4785127.712059225;
        }
        else if (filename.find("synthetic") != std::string::npos) {
            x = 0.0;
            y = 0.0;
        }
        Traj::erp_gpoint.SetXY(x, y);
    }

    std::vector<Traj*> q_set;
    Dataset::Load(filename, n_trajs, min_trajlen, max_trajlen, 2001, q_set); // TODO

    Timer _timer;
    for (auto q: q_set) {
        std::vector<std::pair<float, unsigned int> > sub_result;
        this->Knn(q, measure, is_gpu, k, c, sub_result);
        // this->Knn_bruteforce(q, measure, k);
        // results.push_back(sub_result);
    }
    double querytime = _timer.Elapse();
    std::cout << "[DITA::Knn]done. #=" << n_trajs << ", @=" << querytime << std::endl;

    for (auto& q: q_set) {
        delete q;
    }
    q_set.clear();
    return querytime;
}
