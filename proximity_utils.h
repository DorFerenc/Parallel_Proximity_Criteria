#pragma once

#include "point.h"

void readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
void writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D);
