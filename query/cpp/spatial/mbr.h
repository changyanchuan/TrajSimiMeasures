#ifndef _MBR_H_
#define _MBR_H_

#include <iostream>
#include "point.h"
#include "traj.h"
#include "edge.h"

class MBR {

public:
    MBR();
    MBR(const float& xlo, const float& xhi, const float& ylo, const float& yhi);
    MBR(const Edge& e);
    MBR(const Traj& t);
    
    void SetMBR(const float& xlo, const float& xhi, const float& ylo, const float& yhi);
    void Extend(const MBR& m);
    void Extend(const float& l);
    void Extend(const float& x, const float& y);
    float Area();
    float UnionArea(MBR* m) const;
    void Reset();
    bool Intersect(const MBR& m) const;
    bool Intersect(const Edge& e) const;
    bool Contain(const Point& p) const;
    bool Contain(const MBR& m) const;
    float PointToMBRDistance(const Point& p) const;
    float TrajToMBRDistance(const Traj& t) const;

    friend std::ostream& operator<<(std::ostream& out, MBR& mbr);

    float xlo;
    float xhi;
    float ylo;
    float yhi;

};

#endif