#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "myProto.h"

// /**
//  * CUDA kernel to compute the coordinates (x, y) for each point.
//  * Each thread processes one point.
//  *
//  * @param points Array of points to compute coordinates for.
//  * @param numPoints Number of points to process.
//  * @param tValues Array of t values for coordinate computation.
//  */
// __global__ void computeCoordinatesKernel(Point* points, int numPoints, double* tValues, int tCount) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < numPoints) {
//         Point* p = &points[idx];
//         for (int i = 0; i <= tCount; i++) {
//             double t = tValues[i];
//             p->x[i] = ((p->x2 - p->x1) / 2.0) * sin(t * M_PI / 2.0) + (p->x2 + p->x1) / 2.0;
//             p->y[i] = p->a * p->x[i] + p->b;
//             // p[i].x = ((p->x2 - p->x1) / 2.0) * sin(t * M_PI / 2.0) + (p->x2 + p->x1) / 2.0;
//             // p[i].y = p->a * p[i].x + p->b;;
//         }
//     }
// }

// /**
//  * CUDA kernel to compute the coordinates (x, y) for each point.
//  * Each thread processes one point.
//  *
//  * @param points Array of points to compute coordinates for.
//  * @param numPoints Number of points to process.
//  * @param tValues Array of t values for coordinate computation.
//  * @param tCount Number of t values.
//  */
// __global__ void computeCoordinatesKernel(Point* points, int numPoints, double* tValues, int tCount) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < numPoints) {
//         Point* p = &points[idx];
//         double t = tValues[threadIdx.x]; // Each thread handles a single t value

//         p[idx].x = ((p->x2 - p->x1) / 2.0) * sin(t * M_PI / 2.0) + (p->x2 + p->x1) / 2.0;
//         p[idx].y = p->a * p[idx].x + p->b;
//     }
// }

// __global__ void computeCoordinatesKernel(Point* points, int numPoints, double* tValues, int tCount) {
//     // Calculate thread index
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     // Check if index is within the number of points and tCount
//     if (idx < numPoints && idx < tCount) {
//         Point p = points[idx];
//         double t = tValues[idx];
        
//         // Compute x and y based on the provided formulas
//         p.x = ((p.x2 - p.x1) / 2.0) * sin(t * M_PI / 2.0) + (p.x2 + p.x1) / 2.0;
//         p.y = p.a * p.x + p.b;

//         // Update the point in the global memory
//         points[idx] = p;
//     }
// }

__global__ void computeCoordinatesKernel(Point *points, int numPoints, double *tValues, int tCount) {
  // Get the thread ID.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // If the thread ID is within the range of points to process, compute the coordinates for that point.
  if (idx >= numPoints) return;
  if (idx < 2) {
    // Get the point parameters.
    double x1 = points[idx].x1;
    double x2 = points[idx].x2;
    double a = points[idx].a;
    double b = points[idx].b;

    // Compute the x and y coordinates.
    double t = tValues[idx % tCount];
    double x = ((x2 - x1) / 2) * sin(t * M_PI / 2) + (x2 + x1) / 2;
    double y = a * x + b;

    // Store the coordinates in the point.
    points[idx].x = x;
    points[idx].y = y;
  }
}

// void computeCoordinates(Point *points, int numPoints, double *tValues, int tCount) {
//   // Print the thread ID.
//   printf("Thread ID: %d\n", blockIdx.x * blockDim.x + threadIdx.x);

//   // If the thread ID is within the range of points to process, compute the coordinates for that point.
//   if (threadIdx.x < numPoints) {
//     // Print the point ID.
//     printf("Point ID: %d\n", threadIdx.x);

//     // Get the point parameters.
//     double x1 = points[threadIdx.x].x1;
//     double x2 = points[threadIdx.x].x2;
//     double a = points[threadIdx.x].a;
//     double b = points[threadIdx.x].b;

//     // Compute the x and y coordinates.
//     double t = tValues[threadIdx.x % tCount];
//     double x = ((x2 - x1) / 2) * sin(t * M_PI / 2) + (x2 + x1) / 2;
//     double y = a * x + b;

//     // Print the x and y coordinates.
//     printf("x: %f\n", x);
//     printf("y: %f\n", y);

//     // Store the coordinates in the point.
//     points[threadIdx.x].x = x;
//     points[threadIdx.x].y = y;
//   }
// }





int performGPUComputation(Point* points, int numPoints, double* tValues, int tCount) {
    Point* d_points = NULL;
    cudaError_t cudaStatus;

    // Allocate GPU memory for points
    cudaStatus = cudaMalloc((void**)&d_points, numPoints * sizeof(Point));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error allocating GPU memory: %s\n", cudaGetErrorString(cudaStatus));
        return 0;
    }
    
    // Copy points data from host to device
    cudaStatus = cudaMemcpy(d_points, points, numPoints * sizeof(Point), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying data to GPU: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        return 0;
    }

    // Determine the block size and number of blocks for GPU kernel execution
    int threadsPerBlock = 256; // Number of threads per block
    int numBlocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock; // Calculate number of blocks needed to process all points
    
    printf("Launching GPU kernel...\n"); // TODO df delete this
    printf("numPoints: %d, blockSize: %d\n", numPoints, threadsPerBlock);// TODO df delete this
    printf("Launching GPU kernel with numBlocks: %d and threadsPerBlock: %d ...\n", numBlocks, threadsPerBlock);// TODO df delete this

    // Compute coordinates on GPU using CUDA kernel
    computeCoordinatesKernel<<<numBlocks, threadsPerBlock>>>(d_points, numPoints, tValues, tCount);

    // Check for kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error launching GPU kernel: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        return 0;
    }

    cudaDeviceSynchronize(); // Wait for GPU computations to complete

    // Copy computed data back to host
    cudaStatus = cudaMemcpy(points, d_points, numPoints * sizeof(Point), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Error copying data back from GPU: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_points);
        return 0;
    }

    // Free GPU memory
    cudaFree(d_points);
    printf("GPU computation completed successfully.\n"); // TODO df delete this
    return 1; // Success
}
