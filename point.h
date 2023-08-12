#pragma once

typedef struct {
    int id;
    double x1, x2, a, b;
    double x, y;
} Point;

void computeCoordinates(Point* point, double t);
void testCoordinates(Point* originalPoints, Point* computedPoints, int numPoints, double* tValues, int tCount);