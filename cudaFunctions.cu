#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * CUDA kernel to compute the coordinates (x, y) for each point.
 * Each thread processes one point.
 *
 * @param points Array of points to compute coordinates with.
 * @param numPoints Number of points to process.
 * @param tValues Array of t values for coordinate computation.
 * @param finalPoints Array of FinalPoint to compute coordinates for.
 */
__global__ void computeCoordinatesKernel(Point* points, int numPoints, double* tValues, int tCount, FinalPoint* finalPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= tCount) {
        double t = (tValues[idx]);
        for (int currentPointIndex = 0; currentPointIndex < numPoints; currentPointIndex++) { //TODO check the <=
            int finalPointsIndex = idx * numPoints + currentPointIndex;             // Calculate the index for finalPoints

            // Get the point parameters.
            double x1 = points[currentPointIndex].x1;
            double x2 = points[currentPointIndex].x2;
            double a = points[currentPointIndex].a;
            double b = points[currentPointIndex].b;

            // Compute the x and y coordinates.
            double x = (((x2 - x1) / 2) * sin(t * M_PI / 2) + (x2 + x1) / 2);
            double y = a * x + b;

            // Store the coordinates in the point.
            finalPoints[finalPointsIndex].x = x;
            finalPoints[finalPointsIndex].y = y;
            finalPoints[finalPointsIndex].id = points[currentPointIndex].id;
            finalPoints[finalPointsIndex].tVal = t;
        }
    }
}

int performGPUComputation(Point* points, int numPoints, double* tValues, int tCount, FinalPoint* finalPoints) {
    Point* d_points = NULL;
    FinalPoint* d_finalPoints = NULL;
    double* d_tValues = NULL; // GPU memory for tValues
    cudaError_t cudaStatus;

    // Allocate GPU memory for points
    cudaStatus = cudaMalloc((void**)&d_points, numPoints * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error allocating GPU memory for points: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }

    // Allocate GPU memory for d_finalPoints
    cudaStatus = cudaMalloc((void**)&d_finalPoints, numPoints * tCount * sizeof(FinalPoint));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error allocating GPU memory for d_finalPoints: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        return 0;
    }

    // Allocate GPU memory for tValues
    cudaStatus = cudaMalloc((void**)&d_tValues, (tCount + 1) * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error allocating GPU memory for tValues: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_finalPoints);
        return 0;
    }
    
    // Copy points data from host to device
    cudaStatus = cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying points data to GPU: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_finalPoints);
        cudaFree(d_tValues);
        return 0;
    }
        
    // Copy points data from host to device
    cudaStatus = cudaMemcpy(d_finalPoints, finalPoints, numPoints * tCount *  sizeof(FinalPoint), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying d_finalPoints data to GPU: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_finalPoints);
        cudaFree(d_tValues);
        return 0;
    }

    // Copy tValues from host to device
    cudaStatus = cudaMemcpy(d_tValues, tValues, (tCount + 1) * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying data to GPU for tValues: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_tValues);
        cudaFree(d_finalPoints);
        return 0;
    }

    // Determine the block size and number of blocks for GPU kernel execution
    int threadsPerBlock = 256; // Number of threads per block
    // int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed to process all points
    int numBlocks = (tCount + threadsPerBlock) / threadsPerBlock; // Calculate number of blocks needed to process all points
    
    // printf("Launching GPU kernel...\n"); // TODO df delete this
    // printf("numPoints: %d, blockSize: %d\n", numPoints, threadsPerBlock);// TODO df delete this
    // printf("Launching GPU kernel with numBlocks: %d and threadsPerBlock: %d ...\n", numBlocks, threadsPerBlock);// TODO df delete this

    printf("tValues:\n");
    for (int i = 0; i <= tCount; i++) {
        printf("t[%d] = %lf\n", i, tValues[i]);
    }

    // Compute coordinates on GPU using CUDA kernel
    computeCoordinatesKernel<<<numBlocks, threadsPerBlock>>>(d_points, numPoints, d_tValues, tCount, d_finalPoints);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error launching GPU kernel: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_tValues);
        cudaFree(d_finalPoints);
        return 0;
    }

    cudaDeviceSynchronize(); // Wait for GPU computations to complete

    // Copy computed data back to host
    cudaStatus = cudaMemcpy(finalPoints, d_finalPoints, (numPoints * tCount * sizeof(FinalPoint)), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying data back from GPU: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        cudaFree(d_tValues);
        cudaFree(d_finalPoints);
        return 0;
    }

    // // Copy computed data back to host
    // cudaStatus = cudaMemcpy(points, d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    // if (cudaStatus != cudaSuccess) {
    //     fprintf(stderr, "Error copying data back from GPU: %s\n", cudaGetErrorString(cudaStatus));
    //     cudaFree(d_points);
    //     cudaFree(d_tValues);
    //     cudaFree(d_finalPoints);
    //     return 0;
    // }

    // Free GPU memory
    cudaFree(d_points);
    cudaFree(d_tValues);
    cudaFree(d_finalPoints);
    // printf("GPU computation completed successfully.\n"); // TODO df delete this
    return 1; // Success
}
