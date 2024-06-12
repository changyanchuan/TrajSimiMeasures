#include <limits>
#include <iostream>
#include <cmath>
#include "mbr.h"
#include "point.h"
#include "traj.h"

MBR::MBR() {
    this->Reset();
}


MBR::MBR(const float& xlo, const float& xhi, const float& ylo, const float& yhi) 
        : xlo(xlo), xhi(xhi), ylo(ylo), yhi(yhi) {
}

MBR::MBR(const Edge& e) {
    if (e.start->x <= e.end->x) {
        this->xlo = e.start->x;
        this->xhi = e.end->x;
    }
    else {
        this->xlo = e.end->x;
        this->xhi = e.start->x;
    }

    if (e.start->y <= e.end->y) {
        this->ylo = e.start->y;
        this->yhi = e.end->y;
    }
    else {
        this->ylo = e.end->y;
        this->yhi = e.start->y;
    }
}

MBR::MBR(const Traj& t) {
    this->Reset();

    for (auto p: t.points) {
        if (p->x < this->xlo) this->xlo = p->x;
        if (p->x > this->xhi) this->xhi = p->x;
        if (p->y < this->ylo) this->ylo = p->y;
        if (p->y > this->yhi) this->yhi = p->y;
    }
}

void MBR::SetMBR(const float& xlo, const float& xhi, const float& ylo, const float& yhi) {
    this->xlo = xlo;
    this->xhi = xhi;
    this->ylo = ylo;
    this->yhi = yhi;
}


void MBR::Extend(const MBR& m) {
    if (this->xlo > m.xlo) {this->xlo = m.xlo;}
    if (this->xhi < m.xhi) {this->xhi = m.xhi;}
    if (this->ylo > m.ylo) {this->ylo = m.ylo;}
    if (this->yhi < m.yhi) {this->yhi = m.yhi;}
}


void MBR::Extend(const float& l) {
    this->xlo -= l;
    this->xhi += l;
    this->ylo -= l;
    this->yhi += l;
}


void MBR::Extend(const float& x, const float& y) {
    this->xlo = x < this->xlo ? x : this->xlo;
    this->xhi = this->xhi < x ? x : this->xhi;
    this->ylo = y < this->ylo ? y : this->ylo;
    this->yhi = this->yhi < y ? y : this->yhi;
}


float MBR::Area() {
    return (xhi - xlo) * (yhi - ylo);
}


float MBR::UnionArea(MBR* m) const {
    double xlo_, ylo_, xhi_, yhi_;
    if (this->xlo > m->xlo) {xlo_ = m->xlo;} else {xlo_ = this->xlo;}
    if (this->xhi < m->xhi) {xhi_ = m->xhi;} else {xhi_ = this->xhi;}
    if (this->ylo > m->ylo) {ylo_ = m->ylo;} else {ylo_ = this->ylo;}
    if (this->yhi < m->yhi) {yhi_ = m->yhi;} else {yhi_ = this->yhi;}
    return (xhi_ - xlo_) * (yhi_ - ylo_);
}


void MBR::Reset() {
    this->SetMBR(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(), 
            std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest());
}


bool MBR::Intersect(const MBR& m) const {
    if (this->xlo > m.xhi || m.xlo > this->xhi
            || this->ylo > m.yhi || m.ylo > this->yhi) {
        return false;
    }
    return true;
}


// To check whether the edge has any intersections with the MBR or in it.
bool MBR::Intersect(const Edge& e) const{
// https://stackoverflow.com/questions/16203760/how-to-check-if-line-segment-intersects-a-rectangle

    // For inside case, it is not necessary to verify both e.get_start() and e.get_end().
    if (this->xlo <= e.start->x && e.start->x <= this->xhi 
            && this->ylo <= e.start->y && e.start->y <= this->yhi) {
        return true;
    }
    
    double lines[4][4] = { // 4 lines of rectange
        {this->xlo, this->ylo, this->xhi, this->ylo},
        {this->xhi, this->ylo, this->xhi, this->yhi},
        {this->xlo, this->ylo, this->xlo, this->yhi},
        // {this->xlo, this->ylo, this->xhi, this->ylo},
        {this->xlo, this->yhi, this->xhi, this->yhi}
        // {this->xhi, this->ylo, this->xhi, this->yhi}
    };

    double p0_x = e.start->x;
    double p0_y = e.start->y;
    double p1_x = e.end->x;
    double p1_y = e.end->y;
    double p2_x, p2_y, p3_x, p3_y;
    double s1_x, s1_y, s2_x, s2_y;
    double s, t;

    for (int i = 0; i < 4; ++i) {
        p2_x = lines[i][0];
        p2_y = lines[i][1];
        p3_x = lines[i][2];
        p3_y = lines[i][3];

        s1_x = p1_x - p0_x;
        s1_y = p1_y - p0_y;
        s2_x = p3_x - p2_x;
        s2_y = p3_y - p2_y;

        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y);
        t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y);

        if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
            return true;
        }
    }
    return false; 
}

bool MBR::Contain(const Point& p) const {

    if ( (this->xlo <= p.x && p.x <= this->xhi)
            && (this->ylo <= p.y && p.y <= this->yhi) ) {
        return true;
    }
    return false;
}


bool MBR::Contain(const MBR& m) const {
    if (this->xlo <= m.xlo && m.xhi <= this->xhi 
            && this->ylo <= m.ylo && m.yhi <= this->yhi) {
        return true;
    }
    return false;
}


//https://stackoverflow.com/questions/5254838/calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
float MBR::PointToMBRDistance(const Point& p) const {
    float nearestx = std::max(this->xlo, std::min(p.x, this->xhi));
    float nearesty = std::max(this->ylo, std::min(p.y, this->yhi));
    float dx = p.x - nearestx;
    float dy = p.y - nearesty;
    return sqrt(dx * dx + dy * dy);
}


float MBR::TrajToMBRDistance(const Traj& t) const {
    float mindist = std::numeric_limits<float>::max();
    for (int i = 0; i < t.NumPoints(); ++i) {
        float dist = this->PointToMBRDistance( *(t.points[i]) );
        if (dist < mindist) {
            mindist = dist;
        }
    }
    return mindist;
}

std::ostream& operator<<(std::ostream& out, MBR& mbr) {
    out.precision(9);
    out << "[" << mbr.xlo << "," << mbr.xhi << "," << mbr.ylo << "," << mbr.yhi << "]";
    return out;
}
