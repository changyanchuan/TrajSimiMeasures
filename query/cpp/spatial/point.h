#ifndef _POINT_H_
#define _POINT_H_

#include <vector>

class Point {

public:
    Point(const float& x = 0.0, const float& y = 0.0);

    void SetXY(const float& x, const float& y);

    bool Equal(const Point& p) const;

    Point& operator=(const Point& p);

    float x;
    float y;
};

typedef std::vector<Point*> Points;

#endif