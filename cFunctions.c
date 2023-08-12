#include <stdio.h>
#include <stdlib.h>

#include "point.h"
#include "proximity_utils.h"

/**
 * Test the computed coordinates against expected coordinates for each point and t value.
 *
 * @param originalPoints Array of original points.
 * @param computedPoints Array of computed points.
 * @param numPoints Number of points in the arrays.
 * @param tValues Array of t values for coordinate computation.
 * @param tCount Number of t values.
 */
void testCoordinates(Point* originalPoints, Point* computedPoints, int numPoints, double* tValues, int tCount) {
    // Iterate through each point
    for (int i = 0; i < numPoints; i++) {
        // Iterate through each t value
        for (int j = 0; j <= tCount; j++) {
            double t = tValues[j];
            double expectedX, expectedY;
            
            // Calculate the expected coordinates using the calculateCoordinates function
            calculateCoordinates(&originalPoints[i], t, &expectedX, &expectedY);
            
            // Compare the computed coordinates with the expected coordinates
            if (computedPoints[i].x != expectedX || computedPoints[i].y != expectedY) {
                fprintf(stderr, "Coordinate test failed for point %d at t index %d\n", i, j);
            }
        }
    }
}
