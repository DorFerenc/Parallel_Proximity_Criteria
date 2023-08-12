#pragma once

#include "point.h"

int performGPUComputation(Point* points, int N, double* tValues);
void testCoordinates(Point* originalPoints, Point* computedPoints, int numPoints, double* tValues, int tCount);