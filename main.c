#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "point.h"
#include "proximity_utils.h"
#include "myProto.h"
#include <cstring>

#define FILENAME "InputSmall.txt"
#define MASTER 0
#define SHOULD_TEST 0

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

    if (size != 2) {
        fprintf(stderr, "This code is designed for two processes.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__); // Abort MPI execution
    }

    int numPointsPerWorker, K, tCount, N;
    double D;
    Point* points;
    double* tValues = NULL;
    double t = 0.0; // Placeholder for t value
    Point* points_orig;

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
            int endIdx = (i == size - 1) ? N - 1 : startIdx + numPointsPerWorker - 1;
            
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
    
    if (SHOULD_TEST) {
        // Allocate memory and copy the original points array for testing
        points_orig = (Point*)malloc(numPointsPerWorker * sizeof(Point));
        if (points_orig == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(tValues); // Free previously allocated memory for tValues
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }
        memcpy(points_orig, points, numPointsPerWorker * sizeof(Point));
    }

    printValues(rank, points, numPointsPerWorker, tValues, tCount); // TODO df delete this
    fprintf(stderr, "rank: %d, tCount: %d, numPointsPerWorker: %d\n", rank, tCount, numPointsPerWorker);

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

    if (SHOULD_TEST)
        testCoordinates(points_orig, points, numPointsPerWorker, tValues, tCount); // Test the computed coordinates against expected coordinates 
    
    SatisfiedInfo satisfiedInfos[tCount + 1]; // Create an array to hold satisfiedInfos
    for (int j = 0; j <= tCount; j++) {
        satisfiedInfos[j].t = -1; // Initialize t value
        for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
            satisfiedInfos[j].satisfiedIndices[k] = -1; // Initialize satisfiedIndices to -1
        }
    }

    fprintf(stderr, "rank: %d, tCount: %d, numPointsPerWorker: %d\n", rank, tCount, numPointsPerWorker);
    for (int i = 0; i < (numPointsPerWorker * tCount); i++) {
        printf("rank: %d workerPointsTcount %d:, id: %d x = %.2f, y = %.2f\n", rank, i, workerPointsTcount[i].id, workerPointsTcount[i].x, workerPointsTcount[i].y);
    }

    // int currentPCPointsFound = 0;
    // // Perform Proximity Criteria Check using OpenMP
    // #pragma omp parallel for shared(points, numPointsPerWorker, K, D, tValues, satisfiedInfos, workerPointsTcount) private(t)
    // for (int j = 0; j <= tCount; j++) {
    //     t = tValues[j];
    //     currentPCPointsFound = 0;

    //     // Iterate through each point and perform Proximity Criteria check
    //     for (int i = 0; i < (numPointsPerWorker * tCount); i++) {
    //         int result = checkProximityCriteria(workerPointsTcount[i], workerPointsTcount, (numPointsPerWorker * tCount), K, D, t);

    //         // Update satisfiedInfos if the current point satisfies Proximity Criteria
    //         if (result) {
    //             if (currentPCPointsFound == 0 && satisfiedInfos[j].satisfiedIndices[currentPCPointsFound] != -1)
    //                 fprintf(stderr, "rank: %d, shit\n", rank);
    //             if (currentPCPointsFound == 0) {
    //                 satisfiedInfos[j].t = t; // Update t value
    //                  #pragma omp critical 
    //                     satisfiedInfos[j].satisfiedIndices[currentPCPointsFound] = workerPointsTcount[i].id; // Insert point's index in the list
    //                     currentPCPointsFound++;
    //             } else {
    //                 if (satisfiedInfos[j].satisfiedIndices[currentPCPointsFound] == -1) {
    //                     for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
    //                         if (satisfiedInfos[j].satisfiedIndices[k] != workerPointsTcount[i].id) {
    //                             #pragma omp critical 
    //                                 satisfiedInfos[j].satisfiedIndices[currentPCPointsFound] = workerPointsTcount[i].id; // Insert point's index in the list
    //                                 currentPCPointsFound++;
    //                         } 
    //                     }
    //                 }
    //             }
    //         }
    //         if (currentPCPointsFound >= MAX_NUM_SATISFIED_POINTS)
    //             break;
    //     }
    // }

    // Perform Parallel Proximity Criteria Check using OpenMP
    #pragma omp parallel for shared(points, numPointsPerWorker, K, D, tValues, satisfiedInfos, workerPointsTcount) private(t)
    for (int j = 0; j <= tCount; j++) {
        t = tValues[j];
        int currentPCPointsFound = 0;
        //int satisfiedIndices[MAX_NUM_SATISFIED_POINTS] = {-1}; // Array to store satisfied indices
        
        // Iterate through each point and perform Proximity Criteria check
        for (int i = 0; i < (numPointsPerWorker * tCount); i++) {
            int result = checkProximityCriteria(workerPointsTcount[i], workerPointsTcount, (numPointsPerWorker * tCount), K, D, t);

            // Update satisfiedIndices if the current point satisfies Proximity Criteria
            #pragma omp critical
            {
                if (result) {
                    // int alreadyAdded = 0;
                    for (int k = 0; k < currentPCPointsFound; k++) {
                        if (satisfiedInfos[j].satisfiedIndices[k] == workerPointsTcount[i].id) {
                            //alreadyAdded = 1;
                            break;
                        } else {
                            satisfiedInfos[j].t = t;
                            satisfiedInfos[j].satisfiedIndices[currentPCPointsFound] = workerPointsTcount[i].id;
                            currentPCPointsFound++;
                        }
                    }
                }
            }
            
            if (currentPCPointsFound >= MAX_NUM_SATISFIED_POINTS) {
                break; // Three points satisfying Proximity Criteria found, exit loop
            }
        }
        
        // // Update satisfiedInfos with the found indices
        // #pragma omp critical
        // {
        //     satisfiedInfos[j].t = t;
        //     for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
        //         satisfiedInfos[j].satisfiedIndices[k] = satisfiedIndices[k];
        //     }
        // }
    }

    for (int i = 0; i < (numPointsPerWorker * tCount); i++) {
        printf("rank: %d, satisfiedInfos[%d].t: %lf\n", rank, i, satisfiedInfos[i].t);
        for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
            printf("\t satisfiedInfos[%d].satisfiedIndices[%d]:%d\n", i, k, satisfiedInfos[i].satisfiedIndices[k]);
        }
    }

    
    // Send computed results back to the master using MPI_Send
    if (rank != MASTER)
        MPI_Send(satisfiedInfos, tCount * sizeof(SatisfiedInfo), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);  // Send the satisfiedInfos array to the master
    else {
        SatisfiedInfo** collectedSatisfiedInfos = (SatisfiedInfo**)malloc(size * sizeof(SatisfiedInfo*));
        if (collectedSatisfiedInfos == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(points);
            free(tValues);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        for (int i = 0; i < size; i++) {
            collectedSatisfiedInfos[i] = (SatisfiedInfo*)malloc((tCount + 1) * sizeof(SatisfiedInfo));
            if (collectedSatisfiedInfos[i] == NULL) {
                fprintf(stderr, "Memory allocation error\n");
                // Free previously allocated memory
                for (int j = 0; j < i; j++) {
                    free(collectedSatisfiedInfos[j]);
                }
                free(collectedSatisfiedInfos);
                free(points);
                free(tValues);
                MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
            }
        }

        // Receive satisfiedInfos from worker processes
        for (int i = 1; i < size; i++) {
            MPI_Recv(collectedSatisfiedInfos[i], (tCount + 1) * sizeof(SatisfiedInfo), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
        }

         // Combine results from all processes and write to the output file
        if (!writeResults("Output.txt", collectedSatisfiedInfos, size, N, tValues, tCount)) {
            fprintf(stderr, "Error writing results to output file\n");
            // Free allocated memory
            for (int i = 0; i < size; i++) {
                free(collectedSatisfiedInfos[i]);
            }
            free(collectedSatisfiedInfos);
            free(points);
            free(tValues);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        // Free allocated memory
        for (int i = 0; i < size; i++) {
            free(collectedSatisfiedInfos[i]);
        }
        free(collectedSatisfiedInfos);
    }

    // else {
    //     SatisfiedInfo collectedSatisfiedInfos[size][tCount]; // Create an array to collect satisfiedInfos from workers

    //     // Receive satisfiedInfos from worker processes
    //     for (int i = 1; i < size; i++)
    //         MPI_Recv(&collectedSatisfiedInfos[i], tCount * sizeof(SatisfiedInfo), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);

    //     // Combine results from all processes and write to the output file
    //     if (!writeResults("Output.txt", collectedResults, N, tValues, tCount)) {
    //         fprintf(stderr, "Error writing results to output file\n");
    //         free(points);
    //         free(tValues);
    //         free(collectedResults);
    //         MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
    //     }

    //     free(collectedResults); 
    // }

    // Free allocated memory
    free(workerPointsTcount);
    free(points);
    free(tValues);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
