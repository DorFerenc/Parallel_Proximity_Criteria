#pragma once

#include "point.h"

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
int checkProximityCriteria(Point point, Point* points, int N, int K, double D, double t);
// int writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D);
// int writeResults(const char* filename, Point* points, int N, double* tValues, int tCount);
int writeResults(const char* filename, SatisfiedInfo** satisfiedInfos, int numWorkers, int N, double* tValues, int tCount);
