#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#include "point.h"
#include "proximity_utils.h"
#include "cuda_utils.h"

#define FILENAME "InputZONA.txt"

int main(int argc, char* argv[]) {
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        fprintf(stderr, "This code is designed for two processes.\n");
        MPI_Abort(MPI_COMM_WORLD, __LINE__); // Abort MPI execution
    }

    if (rank == 0) {
        // Master process reads input data and sends work to workers
        int N, K, tCount;
        double D;
        Point* points = NULL;
        double* tValues = NULL;

        // Read input data and exit if it fails
        if (!readInputData(FILENAME, &N, &K, &D, &tCount, &points)) {
            fprintf(stderr, "Error reading input data from %s\n", FILENAME);
            MPI_Abort(MPI_COMM_WORLD, 1); // Abort MPI with failure status
        }

        // Distribute input points and t values to workers
        // Use MPI_Send and MPI_Recv for communication
        
        // Free allocated memory
        free(points);
    } else if (rank == 1) {
        // Worker process
        int N;
        Point* points;
        double t = 0.0; // Placeholder for t value

        // Receive input points and t values from the master
        
        // Perform GPU-accelerated computation using CUDA
        performGPUComputation(points, N, t);

        // Use OpenMP to parallelize Proximity Criteria check
        // Ensure thread safety for accessing shared data structures
        
        // Send computed results back to the master using MPI_Send
        
        // Free allocated memory
        free(points);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
