#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "point.h"
#include "proximity_utils.h"
#include "cuda_utils.h"

#define FILENAME "InputZONA.txt"
#define MASTER 0

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

    int numPointsPerWorker, K, tCount;
    double D;
    Point* points;
    double* tValues = NULL;
    double t = 0.0; // Placeholder for t value

    if (rank == MASTER) {
        // Master process reads input data and sends work to workers
        int N;

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
        int numPointsPerWorker = N / size;
        for (int i = 1; i < size; i++) {
            int startIdx = (i - 1) * numPointsPerWorker;
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

        //MASTER Allocate memory for points to hold the points data
        points = (Point*)malloc(numPointsPerWorker * sizeof(Point));
        if (points == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(tValues); // Free previously allocated memory for tValues
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

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
    
    // Perform GPU-accelerated computation using CUDA
    if (!performGPUComputation(points, numPointsPerWorker, tValues)) {
        fprintf(stderr, "Error performing GPU computation\n");
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
    }  

    // Use OpenMP to parallelize Proximity Criteria check
    #pragma omp parallel for shared(points, numPointsPerWorker, K, D, tValues) private(t)
    for (int i = 0; i < numPointsPerWorker; i++) {
        for (int j = 0; j <= tCount; j++) {
            t = tValues[j];
            // Perform Proximity Criteria check for each point with specific t value
            int result = checkProximityCriteria(points[i], points, numPointsPerWorker, K, D, t);
            // Update results or perform other necessary operations
        }
    }

    // Ensure thread safety for accessing shared data structures
    
    // Send computed results back to the master using MPI_Send
    if (rank != MASTER)
        MPI_Send(points, numPointsPerWorker * sizeof(Point), MPI_BYTE, MASTER, 0, MPI_COMM_WORLD);
    else 
        // Create an array to collect results from workers
        Point* collectedResults = (Point*)malloc(N * sizeof(Point));
        if (collectedResults == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            free(points);
            free(tValues);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        // Collect results from worker processes
        for (int i = 1; i < size; i++) {
            MPI_Recv(&collectedResults[(i - 1) * numPointsPerWorker], numPointsPerWorker * sizeof(Point), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
        }

        // Combine results from all processes and write to the output file
        if (!combineResultsAndWrite(collectedResults, N, tValues, tCount)) {
            fprintf(stderr, "Error writing results to output file\n");
            free(points);
            free(tValues);
            free(collectedResults);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        free(collectedResults); 
        // Master process receives results from all workers
        // for (int i = 1; i < size; i++) {
        //     MPI_Recv(&points[numPointsPerWorker * i], numPointsPerWorker * sizeof(Point), MPI_BYTE, i, 0, MPI_COMM_WORLD, &status);
        // }

        // // Write results to the output file using writeResults function
        // if (!writeResults("Output.txt", points, N)) {
        //     fprintf(stderr, "Error writing results to the output file\n");
        //     free(points);
        //     free(tValues);
        //     MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        // }
    }

    // Free allocated memory
    free(points);
    free(tValues);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
