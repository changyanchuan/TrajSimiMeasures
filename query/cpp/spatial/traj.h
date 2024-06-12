#ifndef __TRAJ_H__
#define __TRAJ_H__

#include <vector>
#include <map>
#include <iostream>
#include <string>

#include "point.h"
#include "edge.h"

class Traj {

public:
    Traj();
    ~Traj();
    
    size_t NumPoints() const;
    size_t NumEdges() const;
    float Length() const;

    float HausdorffDistance(Traj* b);
    float Frechet(Traj* b);
    float DTW(Traj* b);
    float ERP(Traj* b, Point& p);

    static int TrajSimiGPU(std::vector<Traj*> trajs1, std::vector<Traj*> trajs2,
                            const std::string& measure, std::vector<float>& dists);


    friend std::istream& operator>>(std::istream& in, Traj& traj);
    friend std::ostream& operator<<(std::ostream& out, Traj& traj);
    
    unsigned int id;
    Points points;
    Edges edges;
    
    static Point erp_gpoint;


private:
    static float HausdorffDistance_directed(Traj* a, Traj* b, float* a_dist, float* b_dist,
                                            float** mdist, bool rotate_mdist);
};


#endif