#include <random>
#include <algorithm>
#include <utility>
#include <vector>
#include <cmath>
#include <roaring/roaring.hh> // namespace roaring
#include <iostream>
#include <fstream>
#include <string>

#include "DFT.h"
#include "strtree.h"
#include "rtree.h"
#include "edge.h"
#include "tool_funcs.h"
#include "BS_thread_pool.hpp"
#include "dataset.hpp"


DFT::DFT(unsigned short rtree_fanout) {
    this->global_index = nullptr;
    this->rtree_fanout = rtree_fanout;
    this->space_mbr = new MBR();
}


DFT::~DFT() {
    delete this->space_mbr;
    this->space_mbr = nullptr;

    this->vec_local_index.clear();
    delete this->global_index;
    this->global_index = nullptr;

    for (auto& t: this->trajs) {
        delete t;
    }
    this->trajs.clear();
    this->edges.clear();
}


double DFT::Build(const std::string& filename, const int& n_trajs, int min_trajlen, int max_trajlen) {
    // 1. read all trajs
    // 2. create edge list
    // 3. do partitioning
    // 4. use 'partitions' and 'list of edge lists' to create global index
    // 5. create local index by each edge list and link the local index to the global one
    // 6. fill the bitmap of trajids bottom-up. may move this to the rtree??


    // 1.
    Dataset::Load(filename, n_trajs, min_trajlen, max_trajlen, 2000, this->trajs); // TODO
    // this->ReadDataset(filename, n_trajs, this->trajs);
    this->num_trajs = this->trajs.size();

    Timer timer_;

    // 2.
    for (auto t: this->trajs) {
        for (auto e: t->edges) {
            this->edges.push_back(e);
            this->space_mbr->Extend(e->start->x, e->start->y);
            this->space_mbr->Extend(e->end->x, e->end->y);
        }
    }
    this->num_edges = this->edges.size();
    std::cout << "[Dataset]#edges=" << this->edges.size() << std::endl;
    std::cout << "[Dataset]mbr=" << *(this->space_mbr) << std::endl;


    // 3. 4. 5.
    std::vector<std::vector<MBR*> > vec_mbrs;
    std::vector<std::vector<Edges> > vec_edges_in_par;
    this->Partitioning(&(this->edges), this->rtree_fanout, vec_mbrs, vec_edges_in_par);
    this->CreateGlobalLocalIndex(vec_mbrs, vec_edges_in_par);

    // this->CreateSingleIndex(this->edges);
    
    // 6.
    this->global_index->UpdateTrajids(this->global_index->root);

    double buildtime = timer_.Elapse();
    std::cout << "[DFT::Build]done. @=" << buildtime << ", #edges=" << this->num_edges << "/" << this->global_index->size << std::endl;
    return buildtime;
}


int DFT::Knn(Traj* q, const std::string& measure, int is_gpu, 
            unsigned int k, unsigned int c, 
            std::vector<std::pair<float, unsigned int> >& results) {
    // 1. compute epsilon
    //      for each mbr of partion, verify whether it is intersected with q
    //      union all trajids, sample c*k
    //      compute trajsimi distance between trajs and q
    //      sub-sort and select the k-th as epsilon
    // 2. index query - return trajids
    //      extend traj edges with epsilon
    //      do subtraction from the completed set
    // 3. finallization - compute exact similarity
    // 4. select top-k results

    Timer _timer;

    // 1.
    roaring::Roaring trajids_samples;
    for (auto e: q->edges) {
        for (auto i = 0; i < this->global_index->root->Size(); ++i) {
            // if ( e->Intersect(this->global_index->root->mbrs[i]) ) {
            if ( this->global_index->root->mbrs[i]->Intersect(*e) ) {
                trajids_samples |= this->global_index->root->childnodes[i]->trajids;
                // std::cout << i << " " << this->global_index->root->childnodes[i]->trajids.cardinality() << " "
                //             << *(this->global_index->root->mbrs[i]) << " "
                //             << *e << " "
                //             << std::endl;
            }
        }
    }
    // std::cout << *q << std::endl;
    int n_cand = trajids_samples.cardinality(); // TODO: if n_trajs_in_total < c*k 
    auto set_sample_nums = PickSet(n_cand, c*k );
    std::vector<unsigned int> vec_sampled_trajs;
    for (auto i: set_sample_nums) {
        uint32_t trajid;
        trajids_samples.select(i, &trajid);
        vec_sampled_trajs.push_back((unsigned int)trajid);
    }
    
    // trajsimi dist between sampled trajs and q
    std::vector<float> vec_sampled_dist; 
    if (!is_gpu) {
        for (auto& trajid: vec_sampled_trajs) {
            Traj* dtraj = trajs[trajid];
            if (!measure.compare("Hausdorff"))
                vec_sampled_dist.push_back(  dtraj->HausdorffDistance(q)  );
            else if (!measure.compare("Frechet"))
                vec_sampled_dist.push_back(  dtraj->Frechet(q)  );
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
    // std::cout << "epsilon=" << epsilon << std::endl;

    // 2.
    std::vector<MBR*> qmbr;
    for (auto e: q->edges) {
        MBR* embr = new MBR(*e);
        embr->Extend(epsilon);
        qmbr.push_back(embr);
    }
    roaring::Roaring cand_trajids;    
    this->global_index->RangeQuery_pset(qmbr, cand_trajids);
    roaring::Roaring all_trajids;
    all_trajids.addRange(0, this->num_trajs);
    cand_trajids = all_trajids - cand_trajids;

    g_counter.Add(CNT_KNN_TRAJ_RETRIVED, cand_trajids.cardinality());
    // std::cout << "[Knn]tid=" << q->id 
    //             << ",n_samples_in_L1=" << n_cand
    //             << ",epsilon=" << epsilon 
    //             << ",cand_trajids=" << cand_trajids.cardinality() << "\n";

    if (cand_trajids.cardinality() < k) {
        return 0;
    }

    // 3.
    std::vector< std::pair<float, unsigned int> > cand_trajsimi;
    if (!is_gpu) {
        for(auto _trajid: cand_trajids) {
            if (!measure.compare("Hausdorff")) {
                cand_trajsimi.push_back(  std::make_pair(this->trajs[_trajid]->HausdorffDistance(q), _trajid)  );
            }
            else if (!measure.compare("Frechet")) {
                cand_trajsimi.push_back(  std::make_pair(this->trajs[_trajid]->Frechet(q), _trajid)  );
            }
        }
    }
    else {
        std::vector<float> dists;
        std::vector<Traj*> trajs1;
        std::vector<Traj*> trajs2;
        for (auto _trajid: cand_trajids) {
            trajs1.push_back(trajs[_trajid]);
            trajs2.push_back(q);
        }
        Traj::TrajSimiGPU(trajs1, trajs2, measure, dists);
        for (size_t i = 0; i < dists.size(); ++i) {
            auto _dist = dists[i];
            auto _trajid = trajs1[i]->id;
            cand_trajsimi.push_back(  std::make_pair(_dist, _trajid)  );
        }
        dists.clear();
    }

    // 4.
    std::nth_element(cand_trajsimi.begin(), cand_trajsimi.begin() + k-1,
                    cand_trajsimi.end(), cmp_pair_1st<float, unsigned int>);
    std::sort(cand_trajsimi.begin(), cand_trajsimi.begin() + k, cmp_pair_1st<float, unsigned int>);


    // results.assign(cand_trajsimi.begin(), cand_trajsimi.begin() + k);

    std::cout << "[Knn]tid=" << q->id << ",@=" << _timer <<
                        ",n_samples_in_L1=" << n_cand <<
                        ",epsilon=" << epsilon << 
                        ",k-th_dist=" << cand_trajsimi[k-1].first << 
                        ",#traj_retrived=" << cand_trajids.cardinality() << "\n";


    // garbage collection
    for (auto& mbr: qmbr) {
        delete mbr;
    }
    vec_sampled_trajs.clear();
    cand_trajsimi.clear();
    return 0;
}


// on one thread
double DFT::Knn(const std::string& filename, const int& n_trajs, 
            const std::string& measure, int is_gpu, 
            int min_trajlen, int max_trajlen, unsigned int k, unsigned int c,
            std::vector< std::vector<std::pair<float, unsigned int> > >& results) {
    
    std::vector<Traj*> q_set;
    Dataset::Load(filename, n_trajs, min_trajlen, max_trajlen, 2000, q_set); // TODO
    // this->ReadDataset(filename, n_trajs, q_set);
    Timer _timer;

    for (auto q: q_set) {
        std::vector<std::pair<float, unsigned int> > sub_result;
        this->Knn(q, measure, is_gpu, k, c, sub_result);
        // results.push_back(sub_result);
    }
    double querytime = _timer.Elapse();
    std::cout << "[DFT::Knn]done. #=" << n_trajs << ", @=" << querytime << std::endl;

    for (auto& q: q_set) {
        delete q;
    }
    q_set.clear();
    return querytime;
}


int DFT::Knn_parallel(const std::string& filename, const int& n_trajs, 
                        const std::string& measure, unsigned int k, unsigned int c) {
    std::vector<Traj*> q_set;
    Dataset::LoadH5(filename, n_trajs, 20, 200, 2000, q_set); // TODO
    // this->ReadDataset(filename, n_trajs, q_set);
    Timer _timer;
    
    // https://github.com/bshoshany/thread-pool
    BS::thread_pool pool(8);
    for (auto& q: q_set) {
        std::vector<std::pair<float, unsigned int> > sub_result;
        pool.push_task([&] { this->Knn(q, measure, 0, k, c, sub_result); });
    }
    pool.wait_for_tasks();

    std::cout << "[DFT::Knn]done. #=" << n_trajs << ",@=" << _timer << std::endl;

    for (auto& q: q_set) {
        delete q;
    }
    q_set.clear();
    return 0;
}


int DFT::Knn_bruteforce(Traj* q, unsigned int k) {
    Timer _timer;
    auto vec_dist = new std::vector<std::pair<float, unsigned int> >();
    
    for (auto& t: this->trajs) {
        vec_dist->push_back( std::make_pair(q->HausdorffDistance(t), t->id) );
    }

    std::nth_element(vec_dist->begin(), vec_dist->begin() + k-1, vec_dist->end());
    std::sort(vec_dist->begin(), vec_dist->begin() + k);
    std::cout << "[Knn]tid=" << q->id << ",@=" << _timer <<
                        ",k-th_dist=" << (*vec_dist)[k-1].first << "\n";


    delete vec_dist;
    return 0;
}


int DFT::Knn_bruteforce(const std::string& filename, const int& n_trajs, unsigned int k) {
    std::vector<Traj*> q_set;
    Dataset::LoadH5(filename, n_trajs, 20, 200, 2000, q_set); // TODO
    // this->ReadDataset(filename, n_trajs, q_set);
    Timer _timer;

    BS::thread_pool pool(32);
    for (auto& q: q_set) {
        pool.push_task([&] { this->Knn_bruteforce(q, k); });
    }
    pool.wait_for_tasks();


    std::cout << "[DFT::Knn]done. #=" << n_trajs << ",@=" << _timer << std::endl;

    for (auto& q: q_set) {
        delete q;
    }
    q_set.clear();

    return 0;
}


int DFT::ReadDataset(const std::string& filename, const int& n, std::vector<Traj*>& t_set) {
    // assume n <= num_traj_in_file

    std::ifstream fin;
	fin.open(filename, std::ios::in | std::ios::binary);

	if (!fin.is_open()) {
		std::cerr << "error in open file" << std::endl;
		return -1;
	}

    int counter = 0;
    float trajid, npoints;
    float x, y;

    while (  fin.read((char*)&trajid, 4)  ) {
        fin.read((char*)&npoints, 4);

        Traj* traj = new Traj();
        traj->id = (unsigned int) trajid;

        for (size_t i = 0; i < (size_t)npoints; ++i) {
            fin.read((char*)&x, 4);
            fin.read((char*)&y, 4);
            Point* p = new Point(x, y);
            traj->points.push_back(p);
        }
        
        for (size_t i = 0; i < (size_t)npoints - 1; ++i) {
            Edge* e = new Edge(traj->points[i], traj->points[i+1], traj->id);
            traj->edges.push_back(e);
        }
        t_set.push_back(traj);
        
        if (++counter == n) {
            break;
        }
    }
    std::cout << "[DFT::ReadDataset]" << "done. #traj=" << t_set.size() << std::endl;
    fin.close();
    return 0;
}


// return parameters: vec_mbrs, vec_edges_in_par
int DFT::Partitioning(std::vector<Edge*>* edges, const int num_partitions,
                        std::vector<std::vector<MBR*> >& vec_mbrs, 
                        std::vector<std::vector<Edges > >& vec_edges_in_par) {
    // 1. subsampling
    // 2. do partion; 
    //      it is hard to extract it as a str_partion, since here we use sampled edges. 
    //      the obtained mbr of sampled edges cannot cover the whole space. 
    //      hard to define a general return type
    // 3. refine the boundary of all MBRs; Since the x, y splits currently are based
    //      on the sampled points, the boundary of MBRs are not the boundary of all edges.
    // 5. traverse the segments


    int rtn = 0;

    float sampling_rate = 0.01; // TODO: move to the argument list?
    int n_samples = (int)(edges->size() * sampling_rate);
    std::vector<Edge*>* sample_edges = new std::vector<Edge*>();

    // 1.
    std::shuffle(edges->begin(), edges->end(), std::default_random_engine(g_seed));
    sample_edges->assign(edges->begin(), edges->begin() + n_samples); // shallow copy

    // 2.  
    int num_slice = (int) sqrt( (double)num_partitions ); 
    std::vector<float> vec_x_bounding;
    std::vector<std::vector<float> > vec_y_bounding;
    rtn = STRTree::Partition(sample_edges, num_slice, vec_x_bounding, vec_y_bounding); // subspace partition

    // 3.
    vec_x_bounding.front() = this->space_mbr->xlo < vec_x_bounding.front() ? this->space_mbr->xlo : vec_x_bounding.front();
    vec_x_bounding.back() = this->space_mbr->xhi > vec_x_bounding.back() ? this->space_mbr->xhi : vec_x_bounding.back();

    for (auto& vec_y: vec_y_bounding) {
        vec_y.front() = this->space_mbr->ylo < vec_y.front() ? this->space_mbr->ylo : vec_y.front();
        vec_y.back() = this->space_mbr->yhi > vec_y.back() ? this->space_mbr->yhi : vec_y.back();
    }

    // // 4. 
    // // there was an argument, vec_mbrs
    // vec_mbrs.clear();
    // for (int i = 0; i < vec_y_bounding.size(); ++i) {
    //     float xlo = vec_x_bounding[i];
    //     float xhi = vec_x_bounding[i+1];
    //     vec_mbrs.push_back( std::vector<MBR*>() );
    //     for (int j = 0; j < vec_y_bounding[i].size() - 1; ++j) {
    //         MBR* _mbr = new MBR(xlo, xhi, vec_y_bounding[i][j], vec_y_bounding[i][j+1]);
    //         vec_mbrs.back().push_back( _mbr );
    //     }
    // }


    // 5. first x then y
    vec_edges_in_par.clear();
    for (int i = 0; i < vec_y_bounding.size(); ++i) {
        vec_edges_in_par.push_back( std::vector<Edges>() );
        for (int j = 0; j < vec_y_bounding[i].size() - 1; ++j) {
            vec_edges_in_par.back().push_back(Edges());
        }
    }

    for (auto e: *edges) {
        float _x = e->MidPoint().x;
        float _y = e->MidPoint().y;

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
        vec_edges_in_par[_x_idx][_y_idx].push_back(e);
    }

    std::cout << "[DFT::Partitioning]done." << std::endl;
    return rtn;
}


int DFT::CreateGlobalLocalIndex(std::vector<std::vector<MBR*> >& vec_mbrs,
                    std::vector<std::vector<Edges> >& vec_edges_in_par) {

    if (this->global_index) {
        return -1;
    }
    // TODO: assertion of total_num_mbrs <= fanout

    this->global_index = new RTree(this->rtree_fanout);
    this->global_index->root = new RNode(); // alias groot
    RNode* groot = this->global_index->root;
    groot->type = RNodeTypeRoot;
    groot->self_mbr = this->space_mbr;

    for (size_t i = 0; i < vec_edges_in_par.size(); ++i) {
        for (size_t j = 0; j < vec_edges_in_par[i].size(); ++j) {
            // groot->mbrs.push_back( vec_mbrs[i][j] );
            
            RTree* ltree = new RTree(this->rtree_fanout);
            this->vec_local_index.push_back(ltree);

            ltree->Build(vec_edges_in_par[i][j]);
            ltree->root->type = ltree->height == 1 ? RNodeTypeLeaf : RNodeTypeInner;
            ltree->root->parent = groot;

            groot->AddChild(ltree->root, ltree->root->self_mbr);
            this->global_index->size += ltree->size;
            this->global_index->number_of_nodes += ltree->number_of_nodes;
            this->global_index->height = ltree->height + 1 > this->global_index->height ? 
                                        ltree->height + 1 : this->global_index->height;
        }
    }

    groot = nullptr;
    return 0;
}

// local index only.
int DFT::CreateSingleIndex(Edges& vec_edges) {
    if (this->global_index) {
        return -1;
    }

    RTree* ltree = new RTree(this->rtree_fanout);
    ltree->Build(vec_edges);
    this->global_index = ltree;

    return 0;
}


bool DFT::Verify() {
    return this->global_index->Verify();
}