#pragma once

#include "point.h"

int performGPUComputation(Point* points, int N, double* tValues, int tCount, FinalPoint* finalPoints);
void calculateCoordinates(Point* point, double t, double* x, double* y);
void testCoordinates(Point* originalPoints, Point* computedPoints, int numPoints, double* tValues, int tCount);