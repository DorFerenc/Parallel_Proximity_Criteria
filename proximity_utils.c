#include "proximity_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points) {
    /*
    Read input data from the given file and populate the parameters and points array.
    Parameters:
        filename: Name of the input file.
        N: Pointer to store the number of points.
        K: Pointer to store the minimal number of points for Proximity Criteria.
        D: Pointer to store the distance threshold.
        tCount: Pointer to store the number of t values.
        points: Pointer to store the array of points.
    Returns:
        1 on success, 0 on failure.
    */

    FILE* inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        perror("Error opening input file");
        return 0;
    }

    // Read parameters
    fscanf(inputFile, "%d %d %lf %d", N, K, D, tCount);

    // Allocate memory for points array
    *points = (Point*)malloc(*N * sizeof(Point));
    if (*points == NULL) {
        perror("Memory allocation error");
        fclose(inputFile);
        return 0;
    }

    // Read data for each point
    for (int i = 0; i < *N; i++) {
        if (fscanf(inputFile, "%d %lf %lf %lf %lf", &(*points)[i].id, &(*points)[i].x1, &(*points)[i].x2, &(*points)[i].a, &(*points)[i].b) != 5) {
            perror("Error reading point data");
            fclose(inputFile);
            free(*points);
            return 0;
        }
    }

    fclose(inputFile);
    return 1;
}


/**
 * Check Proximity Criteria for a point with a specific t value.
 *
 * @param point The point to check the Proximity Criteria for.
 * @param points Array of points.
 * @param N Number of points in the array.
 * @param K The value of K parameter.
 * @param D The value of D parameter.
 * @param t The specific t value.
 * @return The result of the Proximity Criteria check.
 */
int checkProximityCriteria(Point point, Point* points, int N, int K, double D, double t) {
    int closePoints = 0;

    #pragma omp parallel for reduction(+:closePoints)
    for (int i = 0; i < N; i++) {
        if (i != point.id) {
            double distance = sqrt((point.x - points[i].x) * (point.x - points[i].x) +
                                   (point.y - points[i].y) * (point.y - points[i].y));
            if (distance < D) {
                closePoints++;
            }
        }
    }

    return closePoints >= K;
}

/**
 * Write results to the output file in the specified format.
 *
 * @param filename Name of the output file.
 * @param satisfiedInfos Array of arrays of SatisfiedInfo structs containing information about satisfied points.
 * @param numWorkers Number of worker processes.
 * @param N Total number of points in the dataset.
 * @param tValues Array of t values.
 * @param tCount Number of t values.
 * @return 1 if writing was successful, 0 if there was an error.
 */
int writeResults(const char* filename, SatisfiedInfo** satisfiedInfos, int numWorkers, int N, double* tValues, int tCount) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening output file\n");
        return 0;
    }

    int foundCount = 0; // Counter to keep track of how many sets of 3 points were found

    // Loop through each t value and each worker
    for (int j = 0; j <= tCount; j++) {
        for (int worker = 0; worker < numWorkers; worker++) {
            char message[1024] = ""; // Initialize an empty string message
            int messageLength = 0;

            // Check if the first satisfied index is not -1 (meaning there are satisfied points)
            if (satisfiedInfos[worker][j].satisfiedIndices[0] != -1) {
                messageLength += snprintf(message + messageLength, sizeof(message) - messageLength, "Points");
                int sumIndices = 0;

                // Iterate through the satisfied indices of a specific t and worker
                for (int k = 0; k < MAX_NUM_SATISFIED_POINTS; k++) {
                    int idx = satisfiedInfos[worker][j].satisfiedIndices[k];
                    if (idx != -1) {
                        messageLength += snprintf(message + messageLength, sizeof(message) - messageLength, " pointID%d", idx);
                        sumIndices += idx;
                        // fprintf(file, " pointID%d", idx);
                    }
                }

                // Check if the sum of satisfied indices is not -1 * MAX_NUM_SATISFIED_POINTS
                if (sumIndices != (-1 * MAX_NUM_SATISFIED_POINTS)) {
                    messageLength += snprintf(message + messageLength, sizeof(message) - messageLength, " satisfy Proximity Criteria at t = %.6f\n", satisfiedInfos[worker][j].t);
                    fprintf(file, "%s", message);
                    foundCount++;
                }
            }
        }
    }

    // If no points were found for any t
    if (foundCount == 0) {
        fprintf(file, "There were no %d points found for any t.\n", MAX_NUM_SATISFIED_POINTS);
    }

    fclose(file);
    return 1; // Writing successful
}


// int writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D) {
//     /*
//     Write results to the given output file.
//     Parameters:
//         filename: Name of the output file.
//         tCount: Number of t values.
//         tValues: Array of t values.
//         points: Array of points.
//         N: Number of points.
//         K: Minimal number of points for Proximity Criteria.
//         D: Distance threshold.
//     Returns:
//         1 on success, 0 on failure.
//     */

//     FILE* outputFile = fopen(filename, "w");
//     if (outputFile == NULL) {
//         perror("Error opening output file");
//         exit(1);
//     }

//     // // Write results for points satisfying Proximity Criteria
//     // int found = 0;
//     // for (int i = 0; i <= tCount; i++) {
//     //     // Perform Proximity Criteria check here
        
//     //     if (/* Check if 3 points satisfy Proximity Criteria */) {
//     //         found = 1;
//     //         fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %.2f\n",
//     //                 /* IDs of the points */, tValues[i]);
//     //     }
//     // }

//     // Write results for points satisfying Proximity Criteria
//     int found = 0;
//     for (int i = 0; i <= tCount; i++) {
//         int count = 0;
//         int pointIDs[3] = {-1, -1, -1};

//         // Perform Proximity Criteria check
//         for (int j = 0; j < N; j++) {
//             int closePoints = 0;
//             for (int k = 0; k < N; k++) {
//                 if (j != k) {
//                     double distance = computeDistance(points[j], points[k]);
//                     if (distance < D) {
//                         closePoints++;
//                     }
//                 }
//             }
//             if (closePoints >= K) {
//                 count++;
//                 if (count <= 3) {
//                     pointIDs[count - 1] = points[j].id;
//                 }
//             }
//         }

//         if (count >= 3) {
//             found = 1;
//             fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %.2f\n",
//                     pointIDs[0], pointIDs[1], pointIDs[2], tValues[i]);
//         }
//     }

//     if (!found) {
//         fprintf(outputFile, "There were no 3 points found for any t.\n");
//     }


//     if (!found) {
//         fprintf(outputFile, "There were no 3 points found for any t.\n");
//     }

//     fclose(outputFile);
//     return 1;
// }
