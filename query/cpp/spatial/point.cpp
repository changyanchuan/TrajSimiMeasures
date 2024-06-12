#include "point.h"

Point::Point(const float& x, const float& y) 
        : x(x), y(y) {
}

void Point::SetXY(const float& x, const float& y) {
    this->x = x;
    this->y = y;
}

bool Point::Equal(const Point& p) const {
    if (this->x == p.x && this->y == p.y) {
        return true;
    }
    return false;
}

Point& Point::operator=(const Point& p) {
    if (this != &p) {
        x = p.x;
        y = p.y;
    }
    return *this;
}