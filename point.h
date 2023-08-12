#pragma once

typedef struct {
    int id;
    double x1, x2, a, b;
    double x, y;
} Point;

typedef struct {
    double t; // The t value
    int satisfiedIndices[MAX_NUM_SATISFIED_POINTS]; // Array to hold indices of satisfied points
} SatisfiedInfo;

void computeCoordinates(Point* point, double t);