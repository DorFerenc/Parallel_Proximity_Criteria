#include "proximity_utils.h"
#include <stdio.h>
#include <stdlib.h>

void readInputData(const char* filename, int* N, int* K, double* D, int* tCount, Point** points) {
    /*
    Read input data from the given file and populate the parameters and points array.
    Parameters:
        filename: Name of the input file.
        N: Pointer to store the number of points.
        K: Pointer to store the minimal number of points for Proximity Criteria.
        D: Pointer to store the distance threshold.
        tCount: Pointer to store the number of t values.
        points: Pointer to store the array of points.
    */

    FILE* inputFile = fopen(filename, "r");
    if (inputFile == NULL) {
        perror("Error opening input file");
        exit(1);
    }

    // Read parameters
    fscanf(inputFile, "%d %d %lf %d", N, K, D, tCount);

    // Allocate memory for points array
    *points = (Point*)malloc(*N * sizeof(Point));
    if (*points == NULL) {
        perror("Memory allocation error");
        fclose(inputFile);
        exit(1);
    }

    // Read data for each point
    for (int i = 0; i < *N; i++) {
        if (fscanf(inputFile, "%d %lf %lf %lf %lf", &(*points)[i].id, &(*points)[i].x1, &(*points)[i].x2, &(*points)[i].a, &(*points)[i].b) != 5) {
            perror("Error reading point data");
            fclose(inputFile);
            free(*points);
            exit(1);
        }
    }

    fclose(inputFile);
}

void writeResults(const char* filename, int tCount, double* tValues, Point* points, int N, int K, double D) {
    /*
    Write results to the given output file.
    Parameters:
        filename: Name of the output file.
        tCount: Number of t values.
        tValues: Array of t values.
        points: Array of points.
        N: Number of points.
        K: Minimal number of points for Proximity Criteria.
        D: Distance threshold.
    */

    FILE* outputFile = fopen(filename, "w");
    if (outputFile == NULL) {
        perror("Error opening output file");
        exit(1);
    }

    // // Write results for points satisfying Proximity Criteria
    // int found = 0;
    // for (int i = 0; i <= tCount; i++) {
    //     // Perform Proximity Criteria check here
        
    //     if (/* Check if 3 points satisfy Proximity Criteria */) {
    //         found = 1;
    //         fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %.2f\n",
    //                 /* IDs of the points */, tValues[i]);
    //     }
    // }

    // Write results for points satisfying Proximity Criteria
    int found = 0;
    for (int i = 0; i <= tCount; i++) {
        int count = 0;
        int pointIDs[3] = {-1, -1, -1};

        // Perform Proximity Criteria check
        for (int j = 0; j < N; j++) {
            int closePoints = 0;
            for (int k = 0; k < N; k++) {
                if (j != k) {
                    double distance = computeDistance(points[j], points[k]);
                    if (distance < D) {
                        closePoints++;
                    }
                }
            }
            if (closePoints >= K) {
                count++;
                if (count <= 3) {
                    pointIDs[count - 1] = points[j].id;
                }
            }
        }

        if (count >= 3) {
            found = 1;
            fprintf(outputFile, "Points %d, %d, %d satisfy Proximity Criteria at t = %.2f\n",
                    pointIDs[0], pointIDs[1], pointIDs[2], tValues[i]);
        }
    }

    if (!found) {
        fprintf(outputFile, "There were no 3 points found for any t.\n");
    }


    if (!found) {
        fprintf(outputFile, "There were no 3 points found for any t.\n");
    }

    fclose(outputFile);
}
