#pragma once

#include "point.h"

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
int checkProximityCriteria(FinalPoint point, FinalPoint* points, int N, int K, double D);
int writeResults(const char* filename, SatisfiedInfo* collectedSatisfiedInfos, int size);
