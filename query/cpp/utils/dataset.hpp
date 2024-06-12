#ifndef _DATASET_H_
#define _DATASET_H_

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <highfive/H5File.hpp>
#include <random>

#include "edge.h"
#include "mbr.h"
#include "traj.h"
#include "tool_funcs.h"

template<typename It, typename OutIt>
void SampleReplacement(It b, It e, OutIt o, size_t n) {
    const size_t s = std::distance(b, e);

    for(size_t i = 0; i < n; ++i) {
        It it = b;
        // Move b iterator random number of steps forward.
        int v = std::uniform_int_distribution<>(0, s-1)(g_gen19937);
        std::advance(it, v);
        // Write into output
        *(o++) = *it;
    }
}


class Dataset {

public:
    inline static int Load(const std::string& file_path, const int num_trajs, 
                        const int traj_len_min, const int traj_len_max, const long seed, 
                        std::vector<Traj*>& vec_trajs) {
        if (file_path.find("synthetic") != std::string::npos) {
            return Dataset::LoadSynthetic(num_trajs, traj_len_min, traj_len_max, seed, vec_trajs);
        }
        else {
            return Dataset::LoadH5(file_path, num_trajs, traj_len_min, traj_len_max, seed, vec_trajs);
        } 
    }


    // return: vec_trajs
    inline static int LoadH5(const std::string& file_path, const int num_trajs, 
                        const int traj_len_min, const int traj_len_max, const long seed, 
                        std::vector<Traj*>& vec_trajs) {
        Timer timer_;
        
        // read file
        HighFive::File fh(file_path, HighFive::File::ReadOnly);

        std::vector<double> merc_range;
        fh.getAttribute("merc_range").read(merc_range);

        std::vector<int> trajs_len;
        fh.getDataSet("/trajs_len").read(trajs_len);

        // filtering satisfied trajectories
        std::mt19937 gen(seed);
        std::vector<std::pair<int, int> > indices;
        for (auto it = trajs_len.begin(); it != trajs_len.end(); ++it) {
            int idx = std::distance(trajs_len.begin(), it);
            if (*it >= traj_len_min) {
                if (*it > traj_len_max) {
                    int v = std::uniform_int_distribution<>(traj_len_min, traj_len_max)(gen);
                    indices.push_back(std::make_pair(idx, v));
                }
                else{
                    indices.push_back(std::make_pair(idx, (*it)));
                }
            }
        }
        assert(indices.size() >= sqrt(num_trajs));

        // sample
        std::vector<std::pair<int, int> > sampled_indices;
        SampleReplacement(indices.begin(), indices.end(), std::back_inserter(sampled_indices), num_trajs);

        // create traj dataset
        // vec_trajs
        unsigned int trajid = 0;
        for (auto iv: sampled_indices) {
            std::vector< std::vector<double> > raw_traj;
            fh.getDataSet("/trajs_merc/" + std::to_string(iv.first) ).read(raw_traj);
            
            Traj* traj = new Traj();
            traj->id = trajid;
            size_t n = iv.second;

            for (size_t i = 0; i < n; ++i) {
                Point* p = new Point(raw_traj[i][0], raw_traj[i][1]);
                traj->points.push_back(p);
            }
            
            for (size_t i = 0; i < n - 1; ++i) {
                Edge* e = new Edge(traj->points[i], traj->points[i+1], traj->id);
                traj->edges.push_back(e);
            }
            vec_trajs.push_back(traj);
            trajid += 1;
        }

        trajs_len.clear();
        indices.clear();
        sampled_indices.clear();

        std::cout << "[Dataset::LoadH5]" << "done. @=" << timer_ << ", #traj=" << vec_trajs.size() << std::endl;
        return 0;
    }

    // return: vec_trajs
    inline static int LoadSynthetic(const int num_trajs, const int traj_len_min, 
                        const int traj_len_max, const long seed, 
                        std::vector<Traj*>& vec_trajs) {
        Timer timer_;

        std::mt19937 gen(seed);
        std::uniform_real_distribution<> random01(0, 1);

        unsigned int trajid = 0;
        for (size_t i = 0; i < num_trajs; ++i) {
            int n = std::uniform_int_distribution<>(traj_len_min, traj_len_max)(gen);
            Traj* traj = new Traj();
            traj->id = trajid;
            for (size_t j = 0; j < n; ++j) {
                Point* p = new Point(random01(gen), random01(gen));
                traj->points.push_back(p);
            }

            for (size_t j = 0; j < n - 1; ++j) {
                Edge* e = new Edge(traj->points[j], traj->points[j+1], traj->id);
                traj->edges.push_back(e);
            }
            vec_trajs.push_back(traj);
            trajid += 1;
        }
        std::cout << "[Dataset::LoadSynthetic]" << "done. @=" << timer_ << ", #traj=" << vec_trajs.size() << std::endl;
        return 0;
    }

};




#endif