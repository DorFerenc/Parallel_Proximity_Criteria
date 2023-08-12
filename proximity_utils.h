#pragma once

#include "point.h"

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
int writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D);
