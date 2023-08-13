#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <float.h>

#include "point.h"
#include "proximity_utils.h"
#include "myProto.h"
#include <cstring>

#define FILENAME "InputSmall.txt"
#define MASTER 0

void printValues(int rank, Point* points, int numPointsPerWorker, double* tValues, int tCount) {
    printf("Rank: %d\n", rank);
    
    printf("Points:\n");
    for (int i = 0; i < numPointsPerWorker; i++) {
        printf("rank: %d Point %d: id: %d, x1 = %.2f, x2 = %.2f\n", rank, i, points[i].id, points[i].x1, points[i].x2);
        printf("Point %d: a = %.2f, b = %.2f\n", i, points[i].a, points[i].b);
        printf("Point %d: x = %.2f, y = %.2f\n", i, points[i].x, points[i].y);
        points[i].x = 3.0;
        points[i].y = 6.0;
        printf("CHANGED Point %d: x = %.2f, y = %.2f\n", i, points[i].x, points[i].y);
    }

    printf("tValues:\n");
    for (int i = 0; i <= tCount; i++) {
        printf("t[%d] = %.6f\n", i, tValues[i]);
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    int rank, size;
    MPI_Status  status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size % 8 != 0 && size % 4 != 0 && size % 2 != 0) {
        fprintf(stderr, "This code is designed for amount of process that can be divided by 8 or 4 or 2\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__); // Abort MPI execution
    }

    int numPointsPerWorker, K, tCount, N;
    double D;
    Point* points;
    double* tValues = NULL;
    double t = 0.0; // Placeholder for t value
    FinalPoint* allWorkerPointsTcount;

    if (rank == MASTER) {
        // Master process reads input data and sends work to workers
        // int N;

        // Read input data and exit if it fails
        if (!readInputData(FILENAME, &N, &K, &D, &tCount, &points)) {
            fprintf(stderr, "Error reading input data from %s\n", FILENAME);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        // Generate t values using dynamic memory allocation
        tValues = (double*)malloc((tCount + 1) * sizeof(double));
        if (tValues == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(points);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        for (int i = 0; i <= tCount; i++) {
            tValues[i] = 2.0 * i / tCount - 1;
        }

        // Distribute input points and t values to workers
        numPointsPerWorker = N / size;
        for (int i = 1; i < size; i++) {
            int startIdx = (i) * numPointsPerWorker;
            
            // Send data to worker i
            MPI_Send(&numPointsPerWorker, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&K, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&D, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&tCount, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(tValues, tCount + 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&points[startIdx], numPointsPerWorker * sizeof(Point), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
        // Use MPI_Send and MPI_Recv for communication

    } else { // rank != MASTER
        // Worker process
        // Receive input data and configuration from the master process (rank 0)
        MPI_Recv(&numPointsPerWorker, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&K, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&D, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&tCount, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

        // Allocate memory for tValues to hold t values for the worker
        tValues = (double*)malloc((tCount + 1) * sizeof(double));
        if (tValues == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }
        MPI_Recv(tValues, tCount + 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status); // Receive tValues array from the master process

        // Allocate memory for points to hold the points data for the worker
        points = (Point*)malloc(numPointsPerWorker * sizeof(Point));
        if (points == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(tValues); // Free previously allocated memory for tValues
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }
        MPI_Recv(points, numPointsPerWorker * sizeof(Point), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Receive points array from the master process
    }

    //Both MASTER and WORKERS perform:
    // printValues(rank, points, numPointsPerWorker, tValues, tCount); // TODO df delete this
    // fprintf(stderr, "rank: %d, tCount: %d, numPointsPerWorker: %d\n", rank, tCount, numPointsPerWorker); // TODO df delete this

    // Allocate memory for a new array of points for each worker process
    FinalPoint* workerPointsTcount = (FinalPoint*)malloc(numPointsPerWorker * tCount * sizeof(FinalPoint));
    if (workerPointsTcount == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        free(tValues); // Free previously allocated memory for tValues
        free(points);
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
    }

    // Perform GPU-accelerated computation using CUDA
    if (!performGPUComputation(points, numPointsPerWorker, tValues, tCount, workerPointsTcount)) {
        fprintf(stderr, "Error performing GPU computation\n");
        fprintf(stderr, "numPointsPerWorker: %d, tCount: %d\n", numPointsPerWorker, tCount);
        fprintf(stderr, "RANK: %d, N: %d, size: %d, K: %d\n", rank, N, size, K);
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
    }  

    // BARRIER to make sure everyone knows the all the new calculated points
    if (rank != MASTER) {
        MPI_Send(workerPointsTcount, numPointsPerWorker * tCount * sizeof(FinalPoint), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);
    } else {
        allWorkerPointsTcount = (FinalPoint *)malloc(size * numPointsPerWorker * tCount * sizeof(FinalPoint));
        if (allWorkerPointsTcount == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(tValues); // Free previously allocated memory for tValues
            free(points);
            free(workerPointsTcount);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memcpy(allWorkerPointsTcount, workerPointsTcount, numPointsPerWorker * tCount * sizeof(FinalPoint));
        for (int source = 1; source < size; source++) 
            MPI_Recv((char *)allWorkerPointsTcount + numPointsPerWorker * tCount * source * sizeof(FinalPoint), numPointsPerWorker * tCount * sizeof(FinalPoint), MPI_BYTE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         // Broadcast the merged data to all processes
        MPI_Bcast(allWorkerPointsTcount, size * numPointsPerWorker * tCount * sizeof(FinalPoint), MPI_BYTE, MASTER, MPI_COMM_WORLD);
    }
    if (rank != MASTER) {
        // Receive the merged workerPointsTcount data using broadcast from the master process
        allWorkerPointsTcount = (FinalPoint *)malloc(size * numPointsPerWorker * tCount * sizeof(FinalPoint));
        if (allWorkerPointsTcount == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(tValues); // Free previously allocated memory for tValues
            free(points);
            free(workerPointsTcount);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Bcast(allWorkerPointsTcount, size * numPointsPerWorker * tCount * sizeof(FinalPoint), MPI_BYTE, MASTER, MPI_COMM_WORLD);
    }

    //BOTH MASTER AND WORKERS
    free(workerPointsTcount); // free information not needed
    free(points);
    int myStartIndex = rank * (tCount / size);
    int myEndIndex = (rank + 1) * (tCount / size); // not inclusive (25/50/75/100) not (24/49/74/99)
    int chunkSize = myEndIndex - myStartIndex;
    double numberAllPoints = (size * numPointsPerWorker * tCount);

    // printf("FULL FULL WORKER BLAH\n");
    // for (int i = 0; i < (numberAllPoints); i++) {
    //     printf("rank: %d Point %2d: id: %2d, x= %6.2f, y= %6.2f, tVal: %lf\n", rank, i, allWorkerPointsTcount[i].id, allWorkerPointsTcount[i].x, allWorkerPointsTcount[i].y, allWorkerPointsTcount[i].tVal);
    // }
    
    fprintf(stderr, "Rank %d: myStartIndex = %d, myEndIndex = %d, chunkSize = %d\n", rank, myStartIndex, myEndIndex, chunkSize);// TODO df delete

    // Initialize the satisfiedInfos array
    SatisfiedInfo localSatisfiedInfos[chunkSize]; // Create an array to hold localSatisfiedInfos
    for (int j = 0; j < chunkSize; j++) {
        localSatisfiedInfos[j].t = 0.0; // Initialize t value to maximum double value
        for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++)
            localSatisfiedInfos[j].satisfiedIndices[k] = (-1); // Initialize satisfiedIndices to -1
        localSatisfiedInfos[j].shouldPrint = 0;
    }

    // Allocate memory for a new array of points for each worker process
    FinalPoint* searchPoints = (FinalPoint*)malloc(numPointsPerWorker * size * sizeof(FinalPoint));
    if (searchPoints == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        free(tValues); // Free previously allocated memory for tValues
        free(allWorkerPointsTcount);
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
    }

    // fprintf(stderr, "(numPointsPerWorker * size): %d\n", (numPointsPerWorker * size));
    for (int j = myStartIndex; j < myEndIndex; j++) {
        double currentT = tValues[j];
        localSatisfiedInfos[j - myStartIndex].t = currentT;
        localSatisfiedInfos[j - myStartIndex].shouldPrint = 0;
        int currentSearchPointAmount = 0;
        int currentSatisfiedInfoIndiciesAmount = 0;
        for (int i = 0; i < numberAllPoints; i++) { //find all the points with the current tVal
            if (allWorkerPointsTcount[i].tVal == currentT) {
                searchPoints[currentSearchPointAmount++] = allWorkerPointsTcount[i];
            }
            if (currentSearchPointAmount >= (numPointsPerWorker * size))  
                break;              
        }
        for (int k = 0; k < currentSearchPointAmount; k++) {
            int result = checkProximityCriteria(searchPoints[k], searchPoints, (currentSearchPointAmount), K, D);
            if (result) {
                int shouldADD = 1;
                for (int r = 0; r < MAX_NUM_SATISFIED_POINTS; r++) {
                    if (localSatisfiedInfos[j - myStartIndex].satisfiedIndices[r] == searchPoints[k].id){
                        shouldADD = 0;
                        break;
                    }
                }
                if (shouldADD)
                    localSatisfiedInfos[j - myStartIndex].satisfiedIndices[currentSatisfiedInfoIndiciesAmount++] = searchPoints[k].id;
            }
            if (currentSatisfiedInfoIndiciesAmount >= MAX_NUM_SATISFIED_POINTS)   {
                localSatisfiedInfos[j - myStartIndex].shouldPrint = 1;
                break;
            }
        }
    }

    
    // for (int i = 0; i < chunkSize; i++) {
    //     if (localSatisfiedInfos[i].shouldPrint) {
    //         printf("RANK: %d Points ", rank);
    //         for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
    //             if (localSatisfiedInfos[i].satisfiedIndices[k] != -1) {
    //                 printf("pointID%d ", localSatisfiedInfos[i].satisfiedIndices[k]);
    //             }
    //         }
    //         printf("satisfy Proximity Criteria at t = %.6f\n", localSatisfiedInfos[i].t);
    //     }
    // }

    
    // Send computed results back to the master using MPI_Send
    if (rank != MASTER) {
        MPI_Send(localSatisfiedInfos, chunkSize * sizeof(SatisfiedInfo), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);  // Send the satisfiedInfos array to the master
    }
    else {
        int currentPrintIndex = 0;
        // Initialize collectedSatisfiedInfos shouldPrint to 0
        SatisfiedInfo collectedSatisfiedInfos[size * chunkSize];
        for (int i = 0; i < chunkSize; i++) {
            if (localSatisfiedInfos[i].shouldPrint) 
                collectedSatisfiedInfos[currentPrintIndex++] = localSatisfiedInfos[i];
        }

        // Receive and collect satisfiedInfos from worker processes
        for (int i = 1; i < size; i++) {
            SatisfiedInfo receivedData[chunkSize];
            MPI_Recv(receivedData, chunkSize * sizeof(SatisfiedInfo), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
            for (int j = 0; j < chunkSize; j++)
                if (receivedData[j].shouldPrint) 
                    collectedSatisfiedInfos[currentPrintIndex++] = receivedData[j];
        }

         // Combine results from all processes and write to the output file
        if (!writeResults("Output.txt", collectedSatisfiedInfos, currentPrintIndex)) {
            fprintf(stderr, "Error writing results to output file\n");
            free(allWorkerPointsTcount);
            free(searchPoints);
            free(tValues);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }
    }

    // Free allocated memory
    free(allWorkerPointsTcount);
    free(searchPoints);
    free(tValues);

    // Finalize MPI
    fprintf(stderr, "rank: %d FINISHED\n", rank);
    MPI_Finalize();

    return 0;
}
