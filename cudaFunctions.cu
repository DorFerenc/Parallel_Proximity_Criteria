#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

/**
 * CUDA kernel to compute the coordinates (x, y) for each point.
 * Each thread processes one point.
 *
 * @param points Array of points to compute coordinates for.
 * @param numPoints Number of points to process.
 * @param tValues Array of t values for coordinate computation.
 */
__global__ void computeCoordinatesKernel(Point* points, int numPoints, double* tValues, int tCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numPoints) {
        Point* p = &points[idx];
        for (int i = 0; i <= tCount; i++) {
            double t = tValues[i];
            // p->x[i] = ((p->x2 - p->x1) / 2.0) * sin(t * M_PI / 2.0) + (p->x2 + p->x1) / 2.0;
            // p->y[i] = p->a * p->x[i] + p->b;
            p[i].x = ((p->x2 - p->x1) / 2.0) * sin(t * M_PI / 2.0) + (p->x2 + p->x1) / 2.0;
            p[i].y = p->a * p[i].x + p->b;;
        }
    }
}

int performGPUComputation(Point* points, int numPoints, double* tValues, int tCount) {
    Point* d_points = NULL;

    // Allocate GPU memory for points
    if (cudaMalloc((void**)&d_points, numPoints * sizeof(Point)) != cudaSuccess) {
        return 0;
    }

    // Copy points data from host to device
    if (cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_points);
        return 0;
    }

    // Determine the block size and number of blocks for GPU kernel execution
    int blockSize = 256; // Number of threads per block
    int numBlocks = (numPoints + blockSize - 1) / blockSize; // Calculate number of blocks needed to process all points
    
    // Compute coordinates on GPU using CUDA kernel
    computeCoordinatesKernel<<<numBlocks, blockSize>>>(d_points, numPoints, tValues, tCount);

    // Check for kernel launch errors
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_points);
        return 0;
    }

    // Copy computed data back to host
    if (cudaMemcpy(points, d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(d_points);
        return 0;
    }

    // Free GPU memory
    cudaFree(d_points);

    return 1; // Success
}
