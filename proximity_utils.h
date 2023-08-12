#pragma once

#include "point.h"

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
int checkProximityCriteria(FinalPoint point, FinalPoint* points, int N, int K, double D);
// Function to check if a point ID already exists in an array
int isPointIDAlreadyAdded(int pointID, int *satisfiedIndices, int foundIndices);
// int writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D);
// int writeResults(const char* filename, Point* points, int N, double* tValues, int tCount);
int writeResults(const char* filename, SatisfiedInfo** satisfiedInfos, int numWorkers, int N, double* tValues, int tCount);
