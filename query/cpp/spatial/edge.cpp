#include <cmath>
#include <algorithm>

#include "edge.h"
#include "tool_funcs.h"


Edge::Edge() : start(nullptr), end(nullptr), mid(nullptr), trajid(0) {
}


Edge::Edge(Point* start, Point* end, const unsigned int& trajid) 
        : start(start), end(end), trajid(trajid) {
    this->mid = new Point( (this->start->x + this->end->x) / 2,
                            (this->start->y + this->end->y) / 2 );
}


Edge::~Edge() {
    this->start = nullptr; // not exclusive, dont free here
    this->end = nullptr;
    delete this->mid;
    this->mid = nullptr;
}


bool Edge::Null() const{
    if (start == nullptr || end == nullptr) {
        return false;
    }
    return true;
}


float Edge::Length() const{
    return L2(this->start->x, this->start->y, this->end->x, this->end->y);
}


const Point& Edge::MidPoint() {
    if (this->mid == nullptr) {
        this->mid = new Point( (this->start->x + this->end->x) / 2,
            (this->start->y + this->end->y) / 2 );
    }
    return *(this->mid);
}


std::ostream& operator<<(std::ostream& out, Edge& edge) {
    out.precision(7);
    out << "[" << edge.start->x << "," << edge.start->y << "," << edge.end->x << "," << edge.end->y << "]";
    return out;
}
