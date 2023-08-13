#pragma once

#include "point.h"

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points);
int checkProximityCriteria(FinalPoint point, FinalPoint* points, int N, int K, double D);
int isPointIDAlreadyAdded(int pointID, int *satisfiedIndices, int foundIndices);

int findPointsWithCurrentT(const FinalPoint* allWorkerPointsTcount, int numberAllPoints, double currentT, FinalPoint* searchPoints, int maxSearchPoints);
int checkProximityAndAdd(SatisfiedInfo* satisfiedInfo, FinalPoint* searchPoints, int searchPointAmount, int K, int D);
void processLocalSatisfiedInfos(SatisfiedInfo* localSatisfiedInfos, FinalPoint* allWorkerPointsTcount, int numberAllPoints, double* tValues, int K, int D, int numPointsPerWorker, int size, int myStartIndex, int myEndIndex);
int writeResults(const char* filename, SatisfiedInfo* collectedSatisfiedInfos, int size);
