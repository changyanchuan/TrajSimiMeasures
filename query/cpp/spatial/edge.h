#ifndef _EDGE_H_
#define _EDGE_H_

#include <ostream>
#include <vector>
#include "point.h"

class Edge {
public:
    Edge(); // TODO: remove?
    Edge(Point* start, Point* end, const unsigned int& trajid);
    ~Edge();

    bool Null() const;
    float Length() const;
    const Point& MidPoint();

    float PointToEdgeDistance(Point* p) const;

    friend std::ostream& operator<<(std::ostream& out, Edge& edge);

    Point* start;
    Point* end;
    Point* mid;
    unsigned int trajid;
};

typedef std::vector<Edge*> Edges;

#endif